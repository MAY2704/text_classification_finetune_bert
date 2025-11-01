from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Provided list of extracted keywords from the PDF
extracted_keywords_pdf = [
    'NBX', 'Bank', 'Data', 'Security', 'Policies', 'NBX', 'Bank', 'is', 'committed', 'to', 'maintaining', 'the', 'highest', 'standards', 'of', 'data', 'security', 'and', 'protecting', 'the', 'confidentiality,', 'integrity,', 'and', 'availability', 'of', 'all', 'information', 'entrusted', 'to', 'us', 'by', 'our', 'customers.', 'We', 'recognize', 'the', 'critical', 'importance', 'of', 'safeguarding', 'sensitive', 'data', 'in', 'the', 'financial', 'industry', 'and', 'have', 'implemented', 'robust', 'policies', 'and', 'procedures', 'to', 'mitigate', 'potential', 'risks.', 'Key', 'Data', 'Security', 'Policies', '•', 'Data', 'Encryption:', 'All', 'sensitive', 'data,', 'both', 'in', 'transit', 'and', 'at', 'rest,', 'is', 'protected', 'using', 'strong', 'encryption', 'algorithms.', 'This', 'ensures', 'that', 'unauthorized', 'parties', 'cannot', 'access', 'or', 'decipher', 'the', 'information.', '•', 'Access', 'Controls:', 'Access', 'to', 'customer', 'data', 'is', 'strictly', 'controlled', 'and', 'limited', 'to', 'authorized', 'personnel', 'with', 'a', 'legitimate', 'business', 'need.', 'We', 'employ', 'role', '-based', 'access', 'control', '(RBAC)', 'and', 'multi', '-factor', 'authentication', '(MFA)', 'to', 'verify', 'user', 'identities', 'and', 'prevent', 'unauthorized', 'access.', '•', 'Data', 'Minimization:', 'We', 'collect', 'only', 'the', 'minimum', 'amount', 'of', 'data', 'necessary', 'to', 'provide', 'our', 'services', 'and', 'comply', 'with', 'legal', 'and', 'regulatory', 'requirements.', '•', 'Data', 'Loss', 'Prevention', '(DLP):', 'We', 'utilize', 'DLP', 'tools', 'and', 'techniques', 'to', 'monitor', 'and', 'prevent', 'the', 'unauthorized', 'transfer', 'of', 'sensitive', 'data', 'outside', 'of', 'our', 'secure', 'environment.', '•', 'Regular', 'Security', 'Assessments:', 'We', 'conduct', 'regular', 'security', 'assessments', 'and', 'audits', 'to', 'identify', 'vulnerabilities', 'and', 'ensure', 'the', 'effectiveness', 'of', 'our', 'security', 'controls.', '•', 'Incident', 'Response', 'Plan:', 'We', 'have', 'a', 'comprehensive', 'incident', 'response', 'plan', 'in', 'place', 'to', 'address', 'any', 'data', 'breaches', 'or', 'security', 'incidents', 'in', 'a', 'timely', 'and', 'effective', 'manner.', 'This', 'plan', 'includes', 'procedures', 'for', 'containment,', 'eradication,', 'recovery,', 'and', 'notification.', '•', 'Employee', 'Training:', 'All', 'employees', 'undergo', 'regular', 'training', 'on', 'data', 'security', 'best', 'practices', 'and', 'their', 'responsibilities', 'in', 'protecting', 'customer', 'information.', '•', 'Compliance:', 'NBX', 'Bank', 'complies', 'with', 'all', 'applicable', 'data', 'protection', 'laws', 'and', 'regulations,', 'including', 'but', 'not', 'limited', 'to', 'GDPR.', 'Sensitive', 'Data', 'Attributes', 'In', 'general,', 'the', 'following', 'data', 'attributes', 'are', 'considered', 'sensitive', 'within', 'NBX', 'Bank', 'and', 'are', 'subject', 'to', 'the', 'security', 'policies', 'outlined', 'above:', '•', 'Personal', 'Identifiers:', 'client,', 'name,', 'customer', '•', 'Contact', 'Information:', 'address,', 'phone,', 'email', '•', 'Authentication', 'Data:', 'Secret,', 'private', '•', 'Financial', 'Information:', 'party,', 'counterparty,', 'account', '•', 'Risk', 'Assessment', 'Data:', 'Risk', 'score,', 'Risk', 'rating', '•', 'Location', 'Data:', 'Location', 'This', 'list', 'is', 'not', 'exhaustive,', 'and', 'any', 'data', 'that', 'could', 'be', 'used', 'to', 'identify', 'an', 'individual', 'or', 'compromise', 'their', 'financial', 'security', 'is', 'treated', 'with', 'the', 'utmost', 'care.'
]

# Define a few seed sensitive keywords based on guidelines
seed_sensitive_keywords = [
    "customer",
    "customer sensitive data",
    "client",
    "personal information identifier",
    "credit risk",
    "risk scores"
]

# Load the MiniLM sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 1. Generate Sentence Embeddings for PDF Keywords
pdf_keyword_embeddings = model.encode(extracted_keywords_pdf)
seed_sensitive_embeddings = model.encode(seed_sensitive_keywords)

# 2. Identify 'Sensitive' PDF Keywords using Semantic Similarity
sensitive_pdf_keywords = set()
similarity_threshold = 0.60

for i, pdf_embedding in enumerate(pdf_keyword_embeddings):
    for seed_embedding in seed_sensitive_embeddings:
        similarity = cosine_similarity([pdf_embedding], [seed_embedding])[0][0]
        if similarity >= similarity_threshold:
            sensitive_pdf_keywords.add(extracted_keywords_pdf[i])
            break # Once a keyword is deemed sensitive, no need to compare with other seeds

print("Extracted PDF Keywords:")
print(extracted_keywords_pdf)
print("\nSeed Sensitive Keywords:")
print(seed_sensitive_keywords)
print(f"\nPotentially Sensitive PDF Keywords (based on similarity >= {similarity_threshold}):")
print(sensitive_pdf_keywords)