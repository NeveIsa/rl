ARGS="--popsize=2"
default:
	pixi run python game.py $(ARGS)

deps:
	pixi add $( cat req.txt | tr '\n' ' ')
