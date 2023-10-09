from transformers import pipeline


def sa():
    model = "distilbert-base-uncased-finetuned-sst-2-english"
    # model = "lysandre/dum"
    p = pipeline(task="sentiment-analysis", model=model)

    text = "Love this!"
    response = p(text)
    print(f"text: {text}")
    print(f"model[{model}] response: {response}")


def main():
    sa()

    print("done")


main()
