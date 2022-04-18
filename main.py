from app import app
import uvicorn
import yaml

if __name__=="__main__":
    with open("./config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        uvicorn.run(app, port=config["port"])