PYTHON ?= python3
PYTHONPATH := src

.PHONY: test demo demo-rt demo-both demo-save verify

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) examples/run_sorting_demo.py --viz matplotlib

demo-rt:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) examples/run_sorting_demo.py --viz pybullet

demo-both:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) examples/run_sorting_demo.py --viz both

demo-save:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) examples/run_sorting_demo.py --viz matplotlib --save demo.gif

verify:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v
	MPLCONFIGDIR=/tmp/mpl PYTHONPATH=$(PYTHONPATH) $(PYTHON) examples/run_sorting_demo.py --viz matplotlib --products 8 --save demo.gif
