from transformers import pipeline


def text_classification():
    texts = [
        "This is great",
        "Thanks for nothing",
        "You've got to work on your face",
        "You're beautiful, never change!"
    ]

    models = [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "SamLowe/roberta-base-go_emotions",
    ]

    for m in models:
        classifier = pipeline(task="sentiment-analysis", model=m)
        print(f"\nmodel[{m}] response:")
        for r in zip(texts, classifier(texts)):
            print(f"  {r[0]}: {r[1]}")


def summarization():
    model = "facebook/bart-large-cnn"
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    text = """
    Hugging Face is an AI company that has become a major hub for open-source machine learning. 
    Their platform has 3 major elements which allow users to access and share machine learning resources. 
    First, is their rapidly growing repository of pre-trained open-source machine learning models for things such as natural language processing (NLP), computer vision, and more. 
    Second, is their library of datasets for training machine learning models for almost any task. 
    Third, and finally, is Spaces which is a collection of open-source ML apps.

    The power of these resources is that they are community generated, which leverages all the benefits of open source i.e. cost-free, wide diversity of tools, high quality resources, and rapid pace of innovation. 
    While these make building powerful ML projects more accessible than before, there is another key element of the Hugging Face ecosystem—their Transformers library.
    """
    summarized_text = summarizer(text, min_length=5, max_length=80)[0]['summary_text']
    print(f"model[{model}] summarized text: {summarized_text}")


def main():
    # text_classification()
    summarization()

    print("done")


main()
