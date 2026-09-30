import argparse
import json

from .archive import inspect_model, variable_matrix, native_variable_report, discovered_zm_variables
from .climatology import build_climatology
from .comparison import compare_period
from .config import load_config
from .diagnostics import quicklooks, trend_plots, validate
from .download import download, plan
from .zonal import build_zonal
from .coverage import coverage_model, coverage_matrix, coverage_plot
from .config import model_names, reference_period, models_with_capability, model_period, is_waccmx
from .evaluation import evaluate_fields
from .comparison import _pressure
from .compact import build_compact
import xarray as xr


def _model_type(name):
    if name not in model_names():
        raise argparse.ArgumentTypeError(f"unknown model {name!r}; use jumacs inspect --model all or see config/models/")
    return name


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jumacs")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "download", "zonal", "climatology", "validate", "coverage", "trends", "quicklook"):
        p = sub.add_parser(command)
        p.add_argument("--model", choices=(*model_names(), "all"), required=True)
        if command in ("download", "climatology", "quicklook"):
            p.add_argument("--start-year", type=int)
            p.add_argument("--end-year", type=int)
        if command in ("download", "zonal", "climatology"):
            p.add_argument("--variable", action="append")
        if command == "download":
            p.add_argument("--execute", action="store_true", help="Transfer selected files after showing and saving the plan")
    compare = sub.add_parser("compare")
    compare.add_argument("--models", nargs=2, type=_model_type, default=["GEOSCCM", "EMAC"])
    compare.add_argument("--start-year", type=int)
    compare.add_argument("--end-year", type=int)
    compare.add_argument("--pressure-hpa", nargs="+", type=float, default=[1000, 700, 500, 300, 200, 100, 70, 50, 30, 20, 10, 5, 2, 1])
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--model", choices=model_names(), required=True)
    evaluation.add_argument("--evaluation-file", required=True)
    evaluation.add_argument("--variable", required=True)
    evaluation.add_argument("--start-year", type=int)
    evaluation.add_argument("--end-year", type=int)
    evaluation.add_argument("--pressure-hpa", nargs="+", type=float, default=[1000, 500, 100, 50, 10, 1])
    compact = sub.add_parser("compact", help="Build a compact CCMI to WACCM-X height-grid product")
    compact.add_argument("--model", choices=(*models_with_capability("compact_waccmx"), "all"), required=True)
    compact.add_argument("--start-year", type=int)
    compact.add_argument("--end-year", type=int)
    compact.add_argument("--latitude-step", type=int, default=5)
    compact.add_argument("--altitude-step", type=int, default=1)
    compact.add_argument("--transition-start", type=float, default=55)
    compact.add_argument("--transition-end", type=float, default=65)
    compact.add_argument("--variable", action="append")
    args = parser.parse_args(argv)
    reference = reference_period()["reference_period"]
    if args.command == "compact":
        models = models_with_capability("compact_waccmx") if args.model == "all" else (args.model,)
        start = args.start_year or reference["start_year"]
        end = args.end_year or reference["end_year"]
        for model in models:
            print(build_compact(model, start, end,
                                args.latitude_step, args.altitude_step,
                                args.transition_start, args.transition_end,
                                args.variable))
        return
    if args.command == "compare":
        if args.models[0] == args.models[1]:
            parser.error("select two distinct models")
        print(compare_period(args.start_year or reference["start_year"], args.end_year or reference["end_year"], [p * 100 for p in args.pressure_hpa], tuple(args.models)))
        return
    if args.command == "evaluate":
        from .climatology import variable_product_name
        config = load_config(args.model)
        start, end = args.start_year or reference["start_year"], args.end_year or reference["end_year"]
        name = config["variables"].get(args.variable, args.variable)
        source = config["paths"]["climatology"]
        from .config import ROOT
        model_path = ROOT / source / variable_product_name(args.model, name, start, end)
        with xr.open_dataset(model_path) as model_ds, xr.open_dataset(args.evaluation_file) as evaluation_ds:
            field = evaluation_ds[args.variable]
            model_field = model_ds["mean"].assign_attrs(units=model_ds.attrs["output_units"])
            result = evaluate_fields(model_field, field, _pressure(model_ds), _pressure(evaluation_ds), [p*100 for p in args.pressure_hpa])
            dest = ROOT / "products/comparison/evaluation" / f"jumacs_{args.model.lower()}_{name}_{start}-{end}_evaluation.nc"
            dest.parent.mkdir(parents=True, exist_ok=True)
            result.to_netcdf(dest)
            print(dest)
        return
    models = model_names() if args.model == "all" else (args.model,)
    unresolved = [m for m in models if load_config(m)["model"].get("status", "ready") != "ready" and not (args.command == "inspect")]
    for model in unresolved:
        print(f"{model}: skipped; archive identifiers unresolved in config/models/{model}.yaml")
    models = tuple(m for m in models if m not in unresolved)
    if args.command == "download":
        plans = []
        for model in models:
            period = model_period(model)
            start = args.start_year or period["start_year"]
            end = args.end_year or period["end_year"]
            try:
                payload = plan(model, start, end, args.variable)
            except FileNotFoundError as exc:
                print(f"{model}: skipped; {exc}")
                continue
            plans.append(payload)
            print(json.dumps({key: value for key, value in payload.items() if key != "files"} | {"file_count": len(payload["files"])}, indent=2))
        print(f"Combined selected size estimate: {sum(p['selected_bytes_estimate'] for p in plans):,} bytes")
        if args.execute:
            for payload in plans:
                download(payload)
        return
    for model in models:
        config = load_config(model)
        period = model_period(model)
        start = getattr(args, "start_year", None) or period["start_year"]
        end = getattr(args, "end_year", None) or period["end_year"]
        variables = getattr(args, "variable", None)
        if args.command == "inspect":
            print(json.dumps(inspect_model(model), indent=2))
        elif args.command == "zonal":
            names = variables or sorted(config["variables"].values())
            if is_waccmx(config) and not variables:
                names = sorted(set(names) | discovered_zm_variables(model))
            for name in names:
                try:
                    print(build_zonal(model, name))
                except FileNotFoundError:
                    if variables:
                        raise
        elif args.command == "climatology":
            for path in build_climatology(model, start, end, variables): print(path)
        elif args.command == "validate":
            print(validate(model))
        elif args.command == "coverage":
            print(coverage_model(model))
        elif args.command == "trends":
            for path in trend_plots(model): print(path)
        elif args.command == "quicklook":
            for path in quicklooks(model, start, end): print(path)
    if args.command == "inspect" and args.model == "all":
        print(variable_matrix())
        print(native_variable_report())
    if args.command == "coverage" and args.model == "all":
        print(coverage_matrix())
        print(coverage_plot())


if __name__ == "__main__":
    main()
