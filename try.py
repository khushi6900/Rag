from deepeval.models import OllamaModel

model =  OllamaModel(model="gpt-oss:120b-cloud"
)
print(model.generate("Hello"))