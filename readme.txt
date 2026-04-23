Dr. House
Multimodal AI for Clinical Reasoning and Explainable Diagnosis
Doctors often rely on many different types of information to make a diagnosis including patient symptoms, medical images, lab results and medical history. These data sources are usually analyzed separately which makes diagnosis slower and more difficult.
This project explores the design of AI Dr House, an AI based system that assists doctors by organizing and analyzing multiple types of patient data at the same time. The system takes clinical text, medical images, lab values, and basic patient information as input. Each data type is processed independently using a model suited to that data in order to extract relevant medical information.
The extracted information is then combined into a single structured patient profile. A reasoning agent uses this profile together with medical knowledge such as known symptom and disease relationships to suggest possible diagnoses and explain why each diagnosis was suggested. The system is intended to support clinical decision making rather than replace human judgment.
Goal
The goal of this project is to design and implement a multimodal diagnostic support pipeline that converts heterogeneous patient data into a unified structured representation and uses an AI reasoning agent to generate explainable diagnostic suggestions. 



Project Roadmap

Weeks 1–2: Define Scope & Patient Data Structure
Establish clinical decision-support use case (not automation)
Identify supported data types: clinical text, images, labs, demographics
Design structured patient profile (JSON) and standardize medical entities


Weeks 3–4: Process Clinical Text & Lab Data
Clinical notes: extract symptoms, duration, severity; normalize terminology
Lab results: normalize units, detect abnormalities, map to clinical signals