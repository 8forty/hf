from transformers import pipeline


def sa():
    # model = "distilbert-base-uncased-finetuned-sst-2-english"
    model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    p = pipeline(task="sentiment-analysis", model=model)

    texts = ["This is great",
             "Thanks for nothing",
             "You've got to work on your face",
             "You're beautiful, never change!"
             ]
    responses = p(texts)
    print(f"model[{model}] response:")
    for r in zip(texts, responses):
        print(f"  {r[0]}: {r[1]}")


def main():
    sa()

    print("done")


main()
