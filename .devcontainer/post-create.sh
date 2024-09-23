git config --global --add safe.directory /workspaces/edu-ntnu-idatt2502
gh auth login
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt