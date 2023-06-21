import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

def get_converted_columns(dataframe):
    dictionary={
        "product_id":[],
        "rating":[],
        "rating_count":[]
    }
   
    counter=0 
    for i in range(0,dataframe.shape[0]):
        try:
            pr_id=dataframe["product_id"][i]
            rat=float(dataframe["rating"][i])
            rat_c=int(str(dataframe["rating_count"][i]).replace(",",""))       
            dictionary["product_id"].append(pr_id)
            dictionary["rating"].append(rat)
            dictionary["rating_count"].append(rat_c)
        
        except:
            counter=counter+1

   
    p_dataframe=pd.DataFrame.from_dict(dictionary)
    return p_dataframe

filename="Amazon\\amazon.csv"
data=pd.read_csv(filename)
clean_dataframe=get_converted_columns(data)    

clean_dataframe = clean_dataframe.groupby('product_id').agg({'rating': 'mean', 'rating_count': 'sum'}).reset_index()
clean_dataframe["row"]=clean_dataframe.reset_index().index
# Normalize the ratings
scaler = MinMaxScaler()
clean_dataframe['normalized_rating'] = scaler.fit_transform(clean_dataframe['rating'].values.reshape(-1, 1))#to reshape vazei thn kathe grammi se ypopinka/ypolista
clean_dataframe['normalized_rating_count'] = scaler.fit_transform(clean_dataframe['rating_count'].values.reshape(-1, 1))#to reshape vazei thn kathe grammi se ypopinka/ypolista
# print(clean_dataframe)

# Remove outliers rating
q1 = clean_dataframe['normalized_rating'].quantile(0.25)
q3 = clean_dataframe['normalized_rating'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
# outliers = clean_dataframe[(clean_dataframe['normalized_rating'] < lower_bound) | (clean_dataframe['normalized_rating'] > upper_bound)] #vgalame 74 times (outliers)
clean_dataframe = clean_dataframe[(clean_dataframe['normalized_rating'] >= lower_bound) & (clean_dataframe['normalized_rating'] <= upper_bound)]
# print(len(outliers["product_id"]))
# print(clean_dataframe) - edw exoume 1348 grammes

# Remove outliers rating_count

q1 = clean_dataframe['normalized_rating_count'].quantile(0.25)
q3 = clean_dataframe['normalized_rating_count'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
# outliers = clean_dataframe[(clean_dataframe['normalized_rating'] < lower_bound) | (clean_dataframe['normalized_rating'] > upper_bound)] #vgalame 74 times (outliers)
clean_dataframe = clean_dataframe[(clean_dataframe['normalized_rating_count'] >= lower_bound) & (clean_dataframe['normalized_rating_count'] <= upper_bound)]
# print(len(outliers["product_id"]))
# print(clean_dataframe) - oi grammes ginan 1196

# Prepare the data for clustering
X = clean_dataframe[['normalized_rating', 'normalized_rating_count']]

#elbow method
# # Perform K-means clustering with different values of K
k_values = range(1, 11)  # Range of K values to try
wcss = []  # List to store the WCSS values

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X) #kanei training to model gia na vgalei score kai to kanei gia kathe k
    wcss.append(kmeans.inertia_)  # WCSS value for the current K

# Plot the elbow curve
plt.plot(k_values, wcss, 'bx-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Elbow Method to Find Optimal K')
plt.show()



# Fit the K-means model
k = 3  # Number of clusters from Elbow method
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Get the WCSS (inertia)
wcss = kmeans.inertia_
print("")
print("WCSS:", wcss)
print("")

# # Visualize the clusters
# plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
# plt.xlabel('Rating')
# plt.ylabel('Rating Count')
# plt.title('K-means Clustering')
# plt.show()



# Add the cluster labels to the DataFrame
clean_dataframe['cluster_label'] = kmeans.labels_ #to .labels_ deixnei se poio cluster anhkei to kathe stoixeio
# print(clean_dataframe)

###############################
#B0BPCJM7TB
user_product_id = input("Enter the product ID that you bought: \n")
# Find the cluster label for the given product
user_product_cluster_label = clean_dataframe[clean_dataframe['product_id'] == user_product_id]['cluster_label'].values[0]
# print(user_product_cluster_label)
# Filter the data to include only products in the same cluster as the user input
recommended_products = clean_dataframe[clean_dataframe['cluster_label'] == user_product_cluster_label]
# print(recommended_products)
recommended_products = recommended_products.merge(data[['product_id', 'category']], on='product_id', how='left')#bazo sta recommended products kai to category apo ta data.Stin ousia me auton ton tropo prostheto to kategori se kathe product id twn recommended_products
# print(recommended_products)
user_category = recommended_products[recommended_products['product_id'] == user_product_id]['category'].values[0]
# print(category)

# Print the recommended product IDs
print("")
print("Recommended Products:")
for i in range(0,len(recommended_products['product_id'])):
    product_id=recommended_products["product_id"][i]
    category=recommended_products["category"][i]
    if product_id!=user_product_id and user_category==category :
        print(product_id)