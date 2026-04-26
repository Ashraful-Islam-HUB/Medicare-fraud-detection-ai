import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/sample_claims.csv')

plt.hist(data['claim_amount'])
plt.title("Claim Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")

plt.savefig("results/claim_distribution.png")
plt.show()
