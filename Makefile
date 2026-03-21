PYTHON ?= python3
.PHONY: test demo demo-rt demo-both demo-save verify

test:
	$(PYTHON) -m unittest discover -s tests -v

demo:
	$(PYTHON) examples/run_sorting_demo.py --viz matplotlib

demo-rt:
	$(PYTHON) examples/run_sorting_demo.py --viz pybullet

demo-both:
	$(PYTHON) examples/run_sorting_demo.py --viz both

demo-save:
	$(PYTHON) examples/run_sorting_demo.py --viz matplotlib --save demo.gif

verify:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) examples/run_sorting_demo.py --viz matplotlib --products 8 --save demo.gif
