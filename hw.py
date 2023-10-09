from transformers import pipeline, Conversation


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


def chatbot():
    prompts = [
        "Hi I'm Zoltan, how are you?",
        "Where do you work?",
        "Are you happy?"
    ]

    model = "facebook/blenderbot-400M-distill"
    chat = pipeline(model=model, max_length=100)
    conversation = Conversation()
    for p in prompts:
        conversation.add_user_input(p)
        print(f"user: {p}")
        response = chat(conversation)
        print(f"bot >> {response.generated_responses[-1]}")


def main():
    # text_classification()
    # summarization()

    chatbot()

    print("done")


main()
