.PHONY: aof-test aof-demo
aof-test:
	python -m pytest tests/test_aof.py tests/test_aof_confidence.py tests/test_route_hu.py -q

aof-demo:
	python aof.py "AJo BTN 8bb" --mc 20000 --seed 42 --json
