import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from textblob import TextBlob

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Create a sample DataFrame for the sample Excel file
sample_data = {
    "question": [
        "What is the difference between abstraction and encapsulation?",
        "Explain the concept of polymorphism in object-oriented programming.",
        "Compare and contrast TCP and UDP protocols.",
        "What is a stack data structure, and how is it different from a queue?",
        "Describe the role of a compiler in the software development process."
    ],
    "answer key": [
        "In a comprehensive view, abstraction and encapsulation play pivotal roles in achieving robust and maintainable software architecture. Abstraction not only simplifies the design process by allowing developers to focus on essential aspects but also supports the creation of abstract data types and polymorphism. It promotes a high level of conceptualization by defining common interfaces that guide the interactions between different components. Encapsulation, as a complementary concept, goes beyond information hiding. It fosters the concept of data protection by providing access modifiers that restrict direct access to internal state, allowing controlled manipulation through well-defined methods. This encapsulation of implementation details enhances code security, reduces the risk of unintended side effects, and facilitates future modifications without affecting external code. Together, abstraction and encapsulation form the foundation of object-oriented design principles, fostering code organization, reusability, and adaptability. These principles contribute to the development of scalable and maintainable software systems, making them essential considerations in the field of software engineering.", 
        "Polymorphism encompasses compile-time polymorphism and runtime polymorphism. Compile-time polymorphism, achieved through method overloading, is resolved at compile time. The compiler determines the correct method to be called based on the method signature and parameter types. Runtime polymorphism, facilitated by method overriding, occurs at runtime. The appropriate method implementation is selected dynamically based on the actual type of the object, promoting flexibility and extensibility in code design. Overall, polymorphism is a powerful concept in OOP, supporting the creation of robust and scalable software systems by providing a unified and adaptable interface for objects of diverse types.",
        "In a more comprehensive analysis, it's essential to delve deeper into the use cases, advantages, and disadvantages of TCP and UDP.TCP is well-suited for applications that require a high level of reliability, such as file transfers, email communication, and web browsing. The reliable, ordered delivery of data comes at the cost of increased overhead and latency due to the connection setup and error-checking mechanisms. TCP's flow control ensures that the sender does not overwhelm the receiver with data, making it suitable for scenarios where network conditions can vary. UDP, being connectionless and lacking some of the reliability features of TCP, is favored in situations where speed and low latency are crucial. Real-time applications like online gaming, live streaming, and VoIP benefit from UDP's lightweight nature, as occasional data loss is tolerable, and rapid transmission is prioritized. However, UDP is not suitable for applications where every piece of data must be delivered reliably and in order. In summary, the choice between TCP and UDP depends on the specific requirements of the application. TCP is chosen for applications that prioritize accuracy and completeness, while UDP is preferred for applications that prioritize speed and low latency.",
        "The stack and queue, both integral data structures in computer science, embody contrasting organizational principles that dictate their functionality. A stack, operating on the Last In, First Out (LIFO) principle, manages elements in a manner where the latest addition is the first to be accessed or removed. This streamlined structure supports operations like push (to add an element) and pop (to remove the top element), and its single entry point streamlines the management of recently added items.",
        "In the intricate tapestry of software development, a compiler assumes a multifaceted role that transcends mere code translation. It serves as a linguistic intermediary, translating high-level programming code, authored by developers in languages like Java or C++, into machine-readable binary code. This process, known as compilation, unfolds in multiple stages, including lexical analysis, syntax analysis, optimization, and code generation."
    ],
    "max marks": [5, 5, 5, 5, 5]
}
sample_df = pd.DataFrame(sample_data)

# Save the sample DataFrame to an Excel file
sample_file_path = 'sample_examination_checker.xlsx'
sample_df.to_excel(sample_file_path, index=False)

# Function to load the dataset
@st.cache_data
def load_data(uploaded_file):
    return pd.read_excel(uploaded_file)

# Function to calculate cosine similarity
def calculate_cosine_similarity(student_answer, answer_key):
    vectorizer = TfidfVectorizer()
    tfidf_matrix_student = vectorizer.fit_transform([student_answer])
    tfidf_matrix_answer = vectorizer.transform([answer_key])
    cosine_sim = cosine_similarity(tfidf_matrix_student, tfidf_matrix_answer)
    return cosine_sim[0][0] * 100

# Function to extract keywords
def extract_keywords(answer, max_marks):
    vectorizer = TfidfVectorizer(stop_words='english')
    top_n_mapping = {1: 5, 2: 10, 3: 20, 4: 30, 5: 50}
    top_keywords = top_n_mapping.get(max_marks, 50)
    tfidf_matrix = vectorizer.fit_transform([answer])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    word_tfidf_scores = dict(zip(feature_names, tfidf_scores))
    sorted_words = sorted(word_tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    top_keywords = min(len(sorted_words), top_keywords)
    return [word for word, _ in sorted_words[:top_keywords]]

# Function to calculate grammar accuracy using TextBlob
def calculate_grammar_accuracy(text):
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    num_errors = sum(1 for a, b in zip(text.split(), corrected_text.split()) if a != b)
    total_words = len(text.split())
    accuracy = (total_words - num_errors) / total_words * 100
    return accuracy

# Streamlit UI
st.title("Examination Checker")

# Provide a download link for the sample Excel file
with open(sample_file_path, "rb") as file:
    btn = st.download_button(
        label="Download Sample Excel File",
        data=file,
        file_name="sample_examination_checker.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download-excel",
        help="Click to download a sample Excel file."
    )

# File upload
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    data = load_data(uploaded_file)

    # Select a question from the dataset
    question_index = st.selectbox("Select a question to answer:", data.index)
    question = data.loc[question_index, 'question']
    answer_key = data.loc[question_index, 'answer key']
    max_marks = data.loc[question_index, 'max marks']

    st.write(f"### Question: {question}")

    # Input the answer from the user
    student_answer = st.text_area("Enter your answer here:")

    if st.button("Evaluate Answer"):
        if student_answer:
            # Calculate cosine similarity
            cosine_sim = calculate_cosine_similarity(student_answer, answer_key)
            
            # Extract keywords from student answer
            sa_keywords = extract_keywords(student_answer, max_marks)
            ak_keywords = extract_keywords(answer_key, max_marks)
            
            # Calculate grammar accuracy
            grammar_accuracy = calculate_grammar_accuracy(student_answer)
            
            # Display evaluation results
            st.write(f"**Cosine Similarity:** {cosine_sim:.2f}%")
            st.write(f"**Grammar Accuracy:** {grammar_accuracy:.2f}%")
            st.write("**Keywords in your answer:**", sa_keywords)
            st.write("**Keywords in the answer key:**", ak_keywords)
            
            # Compute final marks (custom logic can be added here)
            final_marks = (cosine_sim * 0.9 + grammar_accuracy * 0.1) * max_marks / 100
            st.write(f"**Final Marks:** {final_marks:.2f} out of {max_marks}")
        else:
            st.write("Please enter an answer to evaluate.")
