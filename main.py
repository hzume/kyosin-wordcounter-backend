import uvicorn
import yaml
from app import app

if __name__=="__main__":
    with open("./config.yaml") as f:
        config = yaml.safe_load(f)
        uvicorn.run(app, port=config["port"])