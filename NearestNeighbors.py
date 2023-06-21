import pandas as pd    #to NearestNeighbors einai supervised learning algorithm (xrisimopoiei kai regration tasks kai classification tasks)
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


def get_converted_columns(dataframe):
    dictionary={
        "product_id":[],
        "rating":[],
        "user_id":[]
    }
   
    counter=0 
    for i in range(0,dataframe.shape[0]):
        try:
            pr_id=dataframe["product_id"][i]
            rat=float(dataframe["rating"][i])
            user_id=dataframe["user_id"][i]      
            dictionary["product_id"].append(pr_id)
            dictionary["rating"].append(rat)
            dictionary["user_id"].append(user_id)
        
        except:
            counter=counter+1

   
    p_dataframe=pd.DataFrame.from_dict(dictionary)
    return p_dataframe

filename="Amazon\\amazon.csv"
data=pd.read_csv(filename)
clean_dataframe=get_converted_columns(data)
# print(clean_dataframe["user_id"][0])

clean_dataframe = clean_dataframe.groupby(['product_id','user_id']).agg({'rating': 'mean'}).reset_index()
clean_dataframe["row"]=clean_dataframe.reset_index().index
# print(clean_dataframe)

# Normalize the ratings
scaler = MinMaxScaler()
clean_dataframe['normalized_rating'] = scaler.fit_transform(clean_dataframe['rating'].values.reshape(-1, 1))#to reshape vazei thn kathe grammi se ypopinka/ypolista
# print(clean_dataframe['rating'].values)
# print(clean_dataframe)

# Remove outliers
q1 = clean_dataframe['normalized_rating'].quantile(0.25)
q3 = clean_dataframe['normalized_rating'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
# outliers = clean_dataframe[(clean_dataframe['normalized_rating'] < lower_bound) | (clean_dataframe['normalized_rating'] > upper_bound)] #vgalame 74 times (outliers)
clean_dataframe = clean_dataframe[(clean_dataframe['normalized_rating'] >= lower_bound) & (clean_dataframe['normalized_rating'] <= upper_bound)]
# print(len(outliers["product_id"]))
# print(clean_dataframe)

# Create the nearest neighbors model
X = clean_dataframe['row'].values.reshape(-1, 1)
y = clean_dataframe['normalized_rating']

model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
model.fit(X, y)#train model


#B09M869Z5V
product_id = input("Enter the product ID that you bought: \n")
product_index = clean_dataframe[clean_dataframe['product_id'] == product_id].index[0]#kratame tis seires pou emfanistike to product ID
# print(product_index)
# print(clean_dataframe.iloc[product_index, :].values.reshape(1, -1))
# distances, indices = model.kneighbors(clean_dataframe.iloc[product_index, :].values.reshape(1, -1))
distances, indices = model.kneighbors([[product_index]])
# print(indices)

# Print the recommendations
# print("Recommendations:")
# for i, index in enumerate(indices[0]):
#     if index != product_index:
#         print(f"{i+1}. Product ID: {clean_dataframe.iloc[index]['product_id']}, Rating: {data.iloc[index]['rating']}")

print("Recommendations:")
for i in range(0,len(indices[0])):
    row=indices[0][i]
    if row != product_index:
        print(f"{i+1}. Product ID: {clean_dataframe['product_id'][row]}, Rating: {clean_dataframe['rating'][row]}")

