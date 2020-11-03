HURST=0.5
test:
	python3 -m pytest

simulation:
	python3 simulation.py
plot:
	python3 pyfbm/pyfbm.py $(HURST)

push:
	@if [ "x$(MSG)" = 'x' ] ; then \
		echo "Usage: MSG='your message here.' make push"; fi
	@test "x$(MSG)" != 'x'
	git commit -a -m "$(MSG)"
	git push
