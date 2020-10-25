HURST=0.5
test:
	python3 -m pytest

simulation:
	python3 simulation.py
plot:
	python3 pyfbm/pyfbm.py $(HURST)
