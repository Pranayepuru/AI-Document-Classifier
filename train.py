import os
import re
import pickle
import json
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Setup NLTK paths and download necessary components
def setup_nltk():
    print("Setting up NLTK resources...")
    for resource in ['stopwords', 'punkt', 'wordnet', 'omw-1.4']:
        try:
            if resource == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif resource == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif resource == 'wordnet':
                nltk.data.find('corpora/wordnet')
            elif resource == 'omw-1.4':
                nltk.data.find('corpora/omw-1.4')
        except LookupError:
            print(f"Downloading NLTK resource: {resource}...")
            nltk.download(resource, quiet=True)
    print("NLTK setup complete.")

setup_nltk()

# Preprocessing Function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Synthetic Dataset Generator
def generate_dataset_if_missing(dataset_dir="dataset"):
    categories = {
        "Resume": [
            "Experienced software engineer with 5 years in python django flask postgresql docker. Managed team of 3 devs.",
            "Junior developer seeking entry-level position. Strong javascript html css frontend skills. Projects in react.",
            "Senior product manager with MBA. Led agile scrum teams to deliver enterprise SaaS applications and cloud products.",
            "Data scientist skilled in machine learning deep learning numpy pandas scikit-learn pytorch. Experience with SQL and Tableau.",
            "DevOps engineer expert in AWS GCP Docker Kubernetes CI/CD pipelines Jenkins Terraform. Passionate about automation.",
            "HR specialist with focus on recruitment, employee relations, onboarding, benefits administration and performance management.",
            "Graphic designer proficient in Adobe Photoshop Illustrator Figma. Portfolio includes UI/UX designs and brand assets.",
            "Systems administrator managing Windows and Linux servers, active directory, network configuration, and security audits.",
            "Marketing manager specializing in digital campaigns, SEO, SEM, content creation, social media growth, and analytics.",
            "Financial analyst with experience in financial modeling, budgeting, forecasting, Excel VBA, and risk assessment."
        ],
        "Invoice": [
            "Invoice number 10243. Bill to Acma Corp. Due date July 2026. Total balance $1250.00. Payment method bank transfer.",
            "Receipt for services. Consulting fees $500. Development work 20 hours at rate $50 per hour. Total amount due $1500.",
            "Billing Statement invoice #88902. Customer ID: 991. Description: Cloud subscription monthly fee. Total cost $99.99.",
            "Invoice IN-2026-004. Date: 2026-06-18. Items: Laptop repair service, memory upgrade. Subtotal: $300. Tax: $24. Grand Total: $324.",
            "Purchase order and invoice invoice-992. Vendor: TechSolutions. Items ordered: 5x office desk chairs. Total amount: $750. Please remit payment.",
            "TAX INVOICE. Sold by RetailGenius. Bill to: John Doe. Item: Smart watch $199. Shipping: $10. Total paid: $209.",
            "Service Invoice. Consultant: Jane Smith. Project: Database optimization. Total hours: 10. Rate: $120/hr. Amount due: $1200.00.",
            "Invoice #099382. Client: Globex. Description of work: Security audit and penetration testing. Total: $5000. Net 30 terms.",
            "Subscription Invoice. Product: Enterprise SaaS License. Billing cycle: Annual. Total Price: $12000. Paid via Credit Card.",
            "Commercial Invoice. Exporter: EuroTech. Importer: USDistributors. Goods: Industrial parts. Total Value: $8500.00."
        ],
        "Report": [
            "Executive summary. Q2 financial performance report. Revenue increased by 15% year-over-year. Market share remains stable.",
            "Annual market analysis report. Competitive analysis shows new entrants in AI space. Recommendations include product pivot.",
            "Project completion report. Key milestones achieved on time. Budget variance is within normal range of 2%.",
            "Operations and efficiency report. Analysis of manufacturing pipeline indicates 5% bottle-neck in packaging division.",
            "Sustainability and environmental impact report 2026. Carbon emissions reduced by 12% through green energy adoption.",
            "Customer satisfaction survey report. Feedback from 1000 users shows 88% approval rating. Areas of improvement: load times.",
            "Q3 sales forecast and strategy report. Target markets identified in Latin America. Sales pipeline analysis.",
            "Annual IT infrastructure audit report. Recommendations for cloud migration and upgrading firewall security.",
            "Employee engagement survey findings. Report details feedback on remote work policies, career growth, and work-life balance.",
            "R&D performance report. Prototype testing phase completed. Patent application filed for new battery technology."
        ],
        "Letter": [
            "Dear Mr. Henderson, I am writing to formally accept the job offer for the position of Senior Analyst. Sincerely, Alice.",
            "Dear Customer, We write to inform you that our terms of service will change starting next month. Best regards, Support Team.",
            "To whom it may concern, I am writing this letter of recommendation for Michael who worked as our intern last summer.",
            "Dear Landlord, This letter serves as a formal 30-day notice of my intent to vacate the apartment. Best regards, Resident.",
            "Dear Board of Directors, Please accept this letter of resignation from my role as VP of Operations. Sincerely, Robert.",
            "Dear Valued Member, Thank you for subscribing to our weekly newsletter. We hope you enjoy the latest updates and tips.",
            "Dear Mayor, I am writing to express my concern regarding the construction noise in our residential neighborhood.",
            "Dear Dr. Watson, Thank you for referring Mr. Holmes to our clinic. We have scheduled his appointment for next Tuesday.",
            "Dear Sarah, Hope you are doing well. I wanted to catch up on our travel plans for the upcoming summer holidays.",
            "To the Admissions Committee, I am writing to express my strong interest in enrolling in your master's program."
        ],
        "Legal": [
            "This confidentiality agreement (Agreement) is entered into between party A and party B. Information shall not be disclosed.",
            "Terms of Service. Under jurisdiction of the State of California. Disputes resolved via binding arbitration in San Francisco.",
            "Intellectual property clause. All code, designs, and documentation created during employment are sole property of the Employer.",
            "Non-disclosure agreement NDA. The parties agree to protect proprietary data and not share secret trade information.",
            "LEASE AGREEMENT. Lessor agrees to rent premises located at 123 Main Street to Lessee for monthly rent of $2000.",
            "INDEMNITY AGREEMENT. Indemnifier agrees to hold harmless and defend Indemnitee from any claims, damages, or liabilities.",
            "PARTNERSHIP AGREEMENT. The partners agree to share profits and losses in proportion to their initial capital contributions.",
            "EMPLOYMENT CONTRACT. This contract defines the rights, duties, and compensation of the Employee under labor laws.",
            "POWER OF ATTORNEY. I hereby appoint my attorney-in-fact to manage, sell, or lease my real estate properties.",
            "SOFTWARE LICENSE AGREEMENT. Licensee is granted a non-exclusive, non-transferable license to use the software program."
        ],
        "Research": [
            "Abstract: This research paper proposes a novel convolutional neural network architecture for image segmentation. Methodology and results.",
            "Introduction: Previous literature has shown significant correlation between sleep patterns and cognitive performance in students.",
            "Discussion and analysis of findings. Experimental results show 98.4% accuracy, outperforming baseline models by a margin.",
            "Literature review. We review historical approaches to distributed consensus protocols, highlighting limitations of Paxos.",
            "Hypothesis: Carbon nanotubes exhibit higher tensile strength when doped with boron. We present structural simulations.",
            "A randomized controlled trial study. Methods: 500 subjects were monitored over a 6-month period to evaluate efficacy.",
            "Quantum entanglement and decoherence in superconducting qubits. We analyze the noise spectrum under cryogenic conditions.",
            "An econometric analysis of inflation rates and interest rates in developing economies. Methodology and regression results.",
            "Global warming impact on marine ecosystems: A metadata analysis of coral reef bleaching events in the Pacific Ocean.",
            "Security vulnerability analysis of smart contract systems on decentralized blockchains. We identify reentrancy exploits."
        ]
    }

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Created dataset directory: {dataset_dir}")

    total_generated = 0
    for category, docs in categories.items():
        cat_dir = os.path.join(dataset_dir, category)
        if not os.path.exists(cat_dir):
            os.makedirs(cat_dir)
        
        # Check if folder is empty
        if len(os.listdir(cat_dir)) == 0:
            for idx, doc_text in enumerate(docs):
                filename = f"{category.lower()}_{idx + 1}.txt"
                filepath = os.path.join(cat_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    # Write a bit of extra boilerplate to make it look like a document
                    f.write(f"DOCUMENT TYPE: {category.upper()}\n")
                    f.write(f"ID: DOC-{category[:3].upper()}-{idx:04d}\n")
                    f.write("="*40 + "\n\n")
                    f.write(doc_text)
                total_generated += 1

    if total_generated > 0:
        print(f"Generated {total_generated} sample text documents in dataset folders.")
    else:
        print("Dataset already contains files. Skipping generation.")

# Training Pipeline
def train_model():
    start_time = time.time()
    
    # 1. Ensure dataset exists
    generate_dataset_if_missing()

    # 2. Load dataset
    dataset_dir = "dataset"
    documents = []
    labels = []

    for category in os.listdir(dataset_dir):
        cat_dir = os.path.join(dataset_dir, category)
        if os.path.isdir(cat_dir):
            for file_name in os.listdir(cat_dir):
                file_path = os.path.join(cat_dir, file_name)
                if os.path.isfile(file_path) and file_name.endswith('.txt'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        documents.append(text)
                        labels.append(category)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    print(f"Loaded {len(documents)} documents across {len(set(labels))} categories.")

    if len(documents) < 10:
        print("Error: Not enough documents in dataset to train models.")
        return

    # 3. Preprocessing
    print("Preprocessing text data...")
    preprocessed_docs = [preprocess_text(doc) for doc in documents]

    # 4. TF-IDF Feature Engineering
    print("Vectorizing text data...")
    # Using character-level and word-level features can improve robustness, but standard word n-grams (1,2) is perfect
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    X = vectorizer.fit_transform(preprocessed_docs)
    y = labels

    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 6. Train Models
    print("Training models...")
    
    # Model 1: Multinomial Naive Bayes
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    
    # Model 2: Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    # Evaluate Models
    def evaluate(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        return acc, precision, recall, f1

    nb_acc, nb_prec, nb_rec, nb_f1 = evaluate(y_test, y_pred_nb)
    lr_acc, lr_prec, lr_rec, lr_f1 = evaluate(y_test, y_pred_lr)

    print(f"\nNaive Bayes Evaluation:")
    print(f"  Accuracy:  {nb_acc:.4f}")
    print(f"  Precision: {nb_prec:.4f}")
    print(f"  Recall:    {nb_rec:.4f}")
    print(f"  F1 Score:  {nb_f1:.4f}")

    print(f"\nLogistic Regression Evaluation:")
    print(f"  Accuracy:  {lr_acc:.4f}")
    print(f"  Precision: {lr_prec:.4f}")
    print(f"  Recall:    {lr_rec:.4f}")
    print(f"  F1 Score:  {lr_f1:.4f}")

    # Determine best model
    best_model_name = "Logistic Regression" if lr_f1 >= nb_f1 else "Naive Bayes"
    best_model = lr_model if lr_f1 >= nb_f1 else nb_model
    best_acc = lr_acc if lr_f1 >= nb_f1 else nb_acc
    best_f1 = lr_f1 if lr_f1 >= nb_f1 else nb_f1

    print(f"\nBest model selected: {best_model_name} (F1 Score: {best_f1:.4f})")

    # Save Best Model and Vectorizer
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("Saved model.pkl and vectorizer.pkl successfully.")

    # Save Stats to a JSON file for the dashboard
    stats = {
        "training_timestamp": time.time(),
        "total_documents": len(documents),
        "categories_count": len(set(labels)),
        "categories": list(set(labels)),
        "best_model": best_model_name,
        "accuracy": round(best_acc * 100, 2),
        "f1_score": round(best_f1 * 100, 2),
        "models_comparison": {
            "NaiveBayes": {
                "accuracy": round(nb_acc * 100, 2),
                "precision": round(nb_prec * 100, 2),
                "recall": round(nb_rec * 100, 2),
                "f1_score": round(nb_f1 * 100, 2)
            },
            "LogisticRegression": {
                "accuracy": round(lr_acc * 100, 2),
                "precision": round(lr_prec * 100, 2),
                "recall": round(lr_rec * 100, 2),
                "f1_score": round(lr_f1 * 100, 2)
            }
        },
        "training_time_seconds": round(time.time() - start_time, 4)
    }

    with open('model_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    print("Saved model_stats.json.")

if __name__ == "__main__":
    train_model()
