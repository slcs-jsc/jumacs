PYTHON ?= python3
CLI = PYTHONPATH="src:$(PYTHONPATH)" $(PYTHON) -m jumacs.cli
MODEL ?= GEOSCCM
START ?= 1985
END ?= 2014

.PHONY: test inspect inspect-all inspect-geosccm inspect-emac inspect-waccmx download zonal climatology compact compare validate coverage trends quicklook site mirror-local mirror-web publish-data

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
	@$(CLI) download --model $(MODEL) --start-year $(START) --end-year $(END) --execute
zonal:
	@$(CLI) zonal --model $(MODEL)
climatology:
	@$(CLI) climatology --model $(MODEL) --start-year $(START) --end-year $(END)
compact:
	@$(CLI) compact --model $(MODEL) --start-year $(START) --end-year $(END)
compare:
	@$(CLI) compare --models GEOSCCM EMAC --start-year $(START) --end-year $(END)
validate:
	@$(CLI) validate --model $(MODEL)
coverage:
	@$(CLI) coverage --model $(MODEL)
trends:
	@$(CLI) trends --model $(MODEL)
quicklook:
	@$(CLI) quicklook --model $(MODEL) --start-year $(START) --end-year $(END)
site:
	@$(PYTHON) scripts/build_index.py
mirror-local:
	@bash scripts/mirror.sh local
mirror-web:
	@bash scripts/mirror.sh web
publish-data:
	@bash scripts/mirror.sh data
