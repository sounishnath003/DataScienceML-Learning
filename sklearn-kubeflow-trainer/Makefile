venv:
	@echo "Activating virtual environment..."
	source ./venv/bin/activate
	echo "Virtual environment activated."

lint:
	clear
	@python3 -m black .

install:
	@python3 -m pip install -r requirements.txt

run: lint
	@echo "Running kubeflow.pipeline..."
	@python -m kubeflow.pipeline
	@echo "Program has finished executing."

