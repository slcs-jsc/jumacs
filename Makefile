PYTHON ?= python3
CLI = PYTHONPATH="src:$(PYTHONPATH)" $(PYTHON) -m jumacs.cli
MODEL ?= GEOSCCM
START ?=
END ?=
PERIOD = $(if $(START),--start-year $(START)) $(if $(END),--end-year $(END))

.PHONY: test inspect inspect-all inspect-geosccm inspect-emac inspect-waccmx download download-plan download-execute zonal climatology compact compare validate coverage trends quicklook site mirror-local mirror-web publish-data

test:
	@PYTHONPATH="src:$(PYTHONPATH)" $(PYTHON) -m pytest -q
inspect:
	@$(CLI) inspect --model $(MODEL)
inspect-all:
	@$(CLI) inspect --model all
inspect-geosccm:
	@$(CLI) inspect --model GEOSCCM
inspect-emac:
	@$(CLI) inspect --model EMAC
inspect-waccmx:
	@$(CLI) inspect --model WACCM-X
download:
	@$(CLI) download --model $(MODEL) $(PERIOD)
download-plan: download
download-execute:
	@$(CLI) download --model $(MODEL) $(PERIOD) --execute
zonal:
	@$(CLI) zonal --model $(MODEL)
climatology:
	@$(CLI) climatology --model $(MODEL) $(PERIOD)
compact:
	@$(CLI) compact --model $(MODEL) $(PERIOD)
compare:
	@$(CLI) compare --models GEOSCCM EMAC $(PERIOD)
validate:
	@$(CLI) validate --model $(MODEL)
coverage:
	@$(CLI) coverage --model $(MODEL)
trends:
	@$(CLI) trends --model $(MODEL)
quicklook:
	@$(CLI) quicklook --model $(MODEL) $(PERIOD)
site:
	@$(PYTHON) scripts/build_index.py
mirror-local:
	@bash scripts/mirror.sh local
mirror-web:
	@bash scripts/mirror.sh web
publish-data:
	@bash scripts/mirror.sh data
