
clear:
	clear

lint: clear
	black .

install: clear
	pip install --upgrade pip black
	pip install -r requirements.txt


run: lint
	export GOOGLE_APPLICATION_CREDENTIALS=service_account.json
	time python main.py --dataset=data/all_tickets_processed_improved_v3.csv

mlflow: lint
	mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri file:ml-runs