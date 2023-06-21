import pandas as pd
# pip install scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor



#synartisi pou tha ta kanei float to rating & integer to rating_count (seira 10)
#shape[0] einai o arithmos grammon 1465 & to shape[1] arithmos stilwn (seira 19)
# print("counter:",counter) (seira 31)
#metatrepo to dictionary se pandas(data frame) (seira 33)

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
# print(clean_dataframe)
# #metirma seiron kai stilon
# print(clean_dataframe.shape)
# #ektipono tis protes seires
# print(data.head())
# #ektipono ta onomata ton stilon
# print("Onomata stilon:")
# print(data.columns)
# print(data["rating_count"])
aggregated_data = clean_dataframe.groupby('product_id').agg({'rating': 'mean', 'rating_count': 'sum'}).reset_index()
aggregated_data["row"]=aggregated_data.reset_index().index
# print(aggregated_data)

X=aggregated_data[['row', 'rating_count']]
y=aggregated_data['rating']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train,y_train) #h synarthsh fit sthn python kanei train to modelo mas


###############################
user_product_id = input("Enter the product ID that you bought: \n")
filtered_df = aggregated_data[aggregated_data['product_id'] == user_product_id]          # print(filtered_df)
row= filtered_df['row'].values[0]                    #pairnw to row gia to product_id pou mou edwse
# {"row":[899]}
dictionary={'row': [row]}
user_input_pandas = pd.DataFrame(dictionary)             #edw metatrepo to dictionary ths python se pandas & pairnoume ton pinaka A ths Join os pandas dataframe
user_input_pandas = user_input_pandas.merge(aggregated_data[['row', 'rating_count']], on='row', how='left')  #me auton ton tropo vrisko to product_id poy eisigage o xrhsths sto Xtest, tha mporousa na to psakso kai sto aggregate_data (left join)

prediction = round(model.predict(user_input_pandas)[0],3)


aggregated_data = aggregated_data.merge(data[['product_id', 'category']], on='product_id', how='left')#bazo sta aggregated kai to category
# category = aggregated_data.loc[aggregated_data['product_id'] == user_product_id, 'category'].values[0]
category = aggregated_data[aggregated_data['product_id'] == user_product_id]['category'].values[0]
perc=0.01
print("")
print("Recomendations:")
for i in range(0,aggregated_data.shape[0]):
    if (aggregated_data['rating'][i]>prediction- prediction*perc)  and (aggregated_data['rating'][i]<prediction+ prediction*perc) and (aggregated_data['category'][i]==category):
        print(aggregated_data["product_id"][i])
# print(prediction)
#B0BMGG6NKT

# print(aggregated_data)
# print(filtered_df)
# print(category)



# 1,7.5,45000
# 2,6,20000
# 1,6,10000
# 1,6.25,55000