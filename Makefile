install:
	pip install --upgrade pip && pip install -r requirements.txt

lint:
	pylint --rcfile=.pylintrc --disable=R app.py 

test:
	python -m pytest -vv --cov=app test_app.py