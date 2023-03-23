import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import evaluate as evaluate
from transformers import get_scheduler
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import T5Config, AutoModelForSequenceClassification, RagTokenizer, RagTokenForGeneration, RagConfig
import matplotlib.pyplot as plt
import os
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

#import plotext as plt
#from torch.utils.tensorboard import SummaryWriter
import argparse
import subprocess


# Set the device      
#device = "mps" if torch.backends.mps.is_available() else "cpu"
#print(f"Using device: {device}")

#un comment later whole block
def print_gpu_memory():
    """
    Print the amount of GPU memory used by the current process
    This is useful for debugging memory issues on the GPU
    """
    # check if gpu is available
    if torch.cuda.is_available():
        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

        p = subprocess.check_output('nvidia-smi')
        print(p.decode("utf-8"))


class BoolQADataset(torch.utils.data.Dataset):
    """
    Dataset for the dataset of BoolQ questions and answers
    """

    def __init__(self, passages, questions, answers, tokenizer, max_len):
        self.passages = passages
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, index):
        """
        This function is called by the DataLoader to get an instance of the data
        :param index:
        :return:
        """

        passage = str(self.passages[index])
        question = self.questions[index]
        answer = self.answers[index]

        #prefix = "question: "
        #suffix = " </s>"

        # The input encoding includes the prefix, question, separator token, passage, and suffix
        #input_encoding = prefix + question + " context: " + passage + suffix


        # this is input encoding for your model. Note, question comes first since we are doing question answering
        # and we don't wnt it to be truncated if the passage is too long
        #input_encoding = question + " [SEP] " + passage

        # encode_plus will encode the input and return a dictionary of tensors
        # encode the input using the T5 tokenizer
        
        encoded_review = self.tokenizer.encode_plus(
            question, #add passage for later case
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length", #padding truncating and attention mask are important
            truncation=True
        )

        label = [1 if answer == "yes" else 0]

        #encoded_label = self.tokenizer.encode_plus(
        #    label,
        #    add_special_tokens=True,
        #    max_length=2,
        #    return_token_type_ids=False,
        #    return_attention_mask=True,
        #    return_tensors="pt",
        #    padding="max_length", #padding truncating and attention mask are important
        #    truncation=True
        #)


        

        return {
            'input_ids': encoded_review['input_ids'][0],  # we only have one example in the batch
            'attention_mask': encoded_review['attention_mask'][0],
            # attention mask tells the model where tokens are padding
            'labels': torch.tensor(label, dtype=torch.long),  # labels are the answers (yes/no)
            #'labelinputid': encoded_label['input_ids'][0],
            #'labelattention': encoded_label['attention_mask'][0]
        }
        
def calculate_accuracy(generated_answers, expected_answers):
    correct_answers = 0
    total_answers = len(generated_answers)

    for generated, expected in zip(generated_answers, expected_answers):
        if generated.lower() == expected.lower():
            correct_answers += 1

    accuracy = correct_answers / total_answers
    return accuracy


def evaluate_model(model, dataloader, tokenizer,device):
    """ Evaluate a PyTorch Model
    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader test_dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on
    :return accuracy
    """
    # load metrics
    #dev_accuracy = evaluate.load('accuracy')

    # turn model into evaluation mode
    model.eval()
    generated_answers = []
    expected_answers = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            generated_answer_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            generated_answer_text = [tokenizer.decode(generated[0], skip_special_tokens=True) for generated in generated_answer_ids]

            generated_answers.extend(generated_answer_text)
            expected_answers.extend(labels)

    accuracy = calculate_accuracy(generated_answers, expected_answers)
    return accuracy 


def train(mymodel, num_epochs, train_dataloader, validation_dataloader, device, lr,trainplot,valplot,mytokenizer):
    """ Train a PyTorch Module

    :param torch.nn.Module mymodel: the model to be trained
    :param int num_epochs: number of epochs to train for
    :param torch.utils.data.DataLoader train_dataloader: DataLoader containing training examples
    :param torch.utils.data.DataLoader validation_dataloader: DataLoader containing validation examples
    :param torch.device device: the device that we'll be training on
    :param float lr: learning rate
    :return None
    """

    # here, we use the AdamW optimizer. Use torch.optim.Adam.
    # instantiate it on the untrained model parameters with a learning rate of 5e-5
    print(" >>>>>>>>  Initializing optimizer")
    optimizer = torch.optim.AdamW(mymodel.parameters(), lr=lr)

    # now, we set up the learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    loss = torch.nn.CrossEntropyLoss()
    accuracylist=[]
    valaccuracylist=[]

    for epoch in range(num_epochs):
        train_generated_answers = []
        train_expected_answers = []

        # put the model in training mode (important that this is done each epoch,
        # since we put the model into eval mode during validation)
        mymodel.train()

        # load metrics
        train_accuracy = evaluate.load('accuracy')

        print(f"Epoch {epoch + 1} training:")

        for i, batch in enumerate(train_dataloader):

            """
            You need to make some changes here to make this function work.
            Specifically, you need to: 
            Extract the input_ids, attention_mask, and labels from the batch; then send them to the device. 
            Then, pass the input_ids and attention_mask to the model to get the logits.
            Then, compute the loss using the logits and the labels.
            Then, call loss.backward() to compute the gradients.
            Then, call optimizer.step()  to update the model parameters.
            Then, call lr_scheduler.step() to update the learning rate.
            Then, call optimizer.zero_grad() to reset the gradients for the next iteration.
            Then, compute the accuracy using the logits and the labels.
            """

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            output = mymodel(input_ids=input_ids, attention_mask=attention_mask)
            predictions = output.logits
            model_loss = loss(predictions.to(device), batch['labels'].to(device))
            generated_answer = mymodel.generate(input_ids=input_ids, attention_mask=attention_mask)
            answer_text = mytokenizer.decode(generated_answer[0], skip_special_tokens=True)
            train_generated_answers.append(answer_text)
            train_expected_answers.append(batch['labels'])


            model_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            #predictions = torch.argmax(predictions, dim=1)

            # update metrics
            #train_accuracy.add_batch(predictions=predictions, references=batch['labels'])

        # print evaluation metrics
        train_accuracy = calculate_accuracy(train_generated_answers, train_expected_answers)
        print(f" ===> Epoch {epoch + 1}")
        print(f" - Average training metrics: accuracy=", train_accuracy)
        accuracylist.append(train_accuracy)


        # normally, validation would be more useful when training for many epochs
        val_accuracy = evaluate_model(mymodel, validation_dataloader, mytokenizer, device)
        print(f" - Average validation metrics: accuracy={val_accuracy}")
        valaccuracylist.append(val_accuracy)

    plt.plot(range(1, num_epochs+1), accuracylist, label="Train plot")
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy over Epochs')
    #plt.savefig(trainplot)
    #plt.show(block=True)
    plt.plot(range(1, num_epochs+1), valaccuracylist,label = "Val plot")
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy over Epochs')
    #plt.show()
    plt.legend()
    plt.savefig(valplot)
    plt.show(block=True)



def pre_process(model_name, batch_size, device, dataset,small_subset=False):
    # download dataset
    print("Loading the dataset ...")
    dataset = load_dataset("boolq")
    dataset = dataset.shuffle()  # shuffle the data

    print("Slicing the data...")
    if small_subset:
        # use this tiny subset for debugging the implementation
        dataset_train_subset = dataset['train'][:10]
        #print(len(dataset_train_subset),'true sub')
        dataset_dev_subset = dataset['train'][:10]
        #print(len(dataset_dev_subset),'true sub')
        dataset_test_subset = dataset['train'][:10]
        #print(len(dataset_test_subset),'true sub')
    else:
        # since the dataset does not come with any validation data,
        # split the training data into "train" and "dev"
        dataset_train_subset = dataset['train'][:500]
        #print(len(dataset_train_subset),'false sub')
        dataset_dev_subset = dataset['validation']
        #print(len(dataset_dev_subset),'false sub')
        dataset_test_subset = dataset['train'][500:]#8000
        #print(len(dataset_test_subset),'false sub')

    # maximum length of the input; any input longer than this will be truncated
    # we had to do some pre-processing on the data to figure what is the length of most instances in the dataset
    max_len = 128 #128 before changed for t5
    #config = T5Config.from_pretrained(model_name)
    #config.max_length = max_len

    print("Loading the tokenizer...")
    mytokenizer = RagTokenizer.from_pretrained('facebook/rag-token-nq')
    #mytokenizer = T5Tokenizer.from_pretrained('t5-small')
    #mytokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loding the data into DS...")
    train_dataset = BoolQADataset(
        passages=list(dataset_train_subset['passage']),
        questions=list(dataset_train_subset['question']),
        answers=list(dataset_train_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )
    validation_dataset = BoolQADataset(
        passages=list(dataset_dev_subset['passage']),
        questions=list(dataset_dev_subset['question']),
        answers=list(dataset_dev_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )
    test_dataset = BoolQADataset(
        passages=list(dataset_test_subset['passage']),
        questions=list(dataset_test_subset['question']),
        answers=list(dataset_test_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )

    print(" >>>>>>>> Initializing the data loaders ... ")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # from Hugging Face (transformers), read their documentation to do this.
    print("Loading the model ...")
    #pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) #could give error
    #config = RagConfig.from_pretrained("facebook/rag-token-nq")
    pretrained_model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")
    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)
    return pretrained_model, train_dataloader, validation_dataloader, test_dataloader,mytokenizer


# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--small_subset", action='store_true')
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")#change back to cuda
    parser.add_argument("--model", type=str, default="facebook/rag-token-nq")
    parser.add_argument("--train_plot", type=str, default="hw7tainplot.png")
    parser.add_argument("--val_plot", type=str, default="hw7valplot.png")
    parser.add_argument("--bar_plot", type=str, default="barplot.png")

    args = parser.parse_args()
    print(f"Specified arguments: {args}")
    assert type(args.small_subset) == bool, "small_subset must be a boolean"
    #added code
    pretrained_model, train_dataloader, validation_dataloader, test_dataloader, tokenizer = pre_process(args.model,
                                                                                                 args.batch_size,
                                                                                               args.device,
                                                                                                args.small_subset)
    
    accuracies=[]
    model_acc={}
    print(" >>>>>>>>  Starting training ... ")
    #train(...)
    train(pretrained_model, args.num_epochs, train_dataloader, validation_dataloader, args.device, args.lr, args.train_plot, args.val_plot,tokenizer)
    # print the GPU memory usage just to make sure things are alright
    print_gpu_memory() 
    val_accuracy = evaluate_model(pretrained_model, validation_dataloader, args.device)
    print(f"Model is {args.model} learning rate{args.lr} epoch is {args.num_epochs}")
    print(f" - Average DEV metrics: accuracy={val_accuracy}")
    test_accuracy = evaluate_model(pretrained_model, test_dataloader, args.device)
    print(f" - Average TEST metrics: accuracy={test_accuracy}")
    model_acc[args.model] = ['Valacc',val_accuracy,'Test acc',test_accuracy]

    print("If model gave out of memory before printing model acc keep that model as 0")
    print("Learning rate and epochs",args.num_epochs,args.lr)
    print("model and accuracies ",model_acc)