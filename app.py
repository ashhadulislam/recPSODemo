import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# PSO Related




class Node():
    def __init__(self, threshold=None, left=None, right=None, info_gain=None, value=None,
                particle=None,min_val=None,max_val=None,samples=None):
        ''' constructor ''' 
        
        # for decision node
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        self.min_val=min_val
        self.max_val=max_val
        
        # for leaf node
        self.value = value
        self.particle=particle
        self.samples=samples
        
        
def gini_index(y):
    ''' 
    Function to compute Gini index 
    
    Parameters:
    y (list or numpy array): List or array containing the class labels
    
    Returns:
    float: Gini index value
    '''
    y=y.reshape(-1)
    total_samples = len(y)
    classes = set(y)
    gini = 0.0
    
    for c in classes:
        proportion = sum([(1 if label == c else 0) for label in y]) / total_samples
        gini += proportion * (1 - proportion)
    
    y=y.reshape([-1,1])
    return gini
        
def entropy(y):
    """
    Calculate the entropy of a label distribution.
    
    Parameters:
    y (np.array): The array containing labels.
    
    Returns:
    float: The entropy value.
    """
    # Get the count of each unique label
    unique, counts = np.unique(y, return_counts=True)
    
    # Calculate the probabilities for each unique label
    probabilities = counts / counts.sum()
    
    # Calculate the entropy
    return -np.sum(probabilities * np.log2(probabilities))

# Your information_gain function
def information_gain(particle, X, y):
    res = multiply_weight(X, particle[0:X.shape[1]])
    threshold = particle[-1]
    X_left, y_left, X_right, y_right = split(X, y, res, threshold)

    parent_entropy = entropy(y)
    left_entropy = entropy(y_left)
    right_entropy = entropy(y_right)
    n = len(y)
    n_left, n_right = len(y_left), len(y_right)

    child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
    return parent_entropy - child_entropy


def information_gain_gini(parent_y, left_y, right_y):
    '''
    Compute information gain at a node of a decision tree using Gini index.
    
    Parameters:
    parent_y (array-like): Class labels at the parent node.
    left_y (array-like): Class labels at the left child node.
    right_y (array-like): Class labels at the right child node.
    
    Returns:
    float: Information gain value.
    '''
    parent_gini = gini_index(parent_y)
    left_weight = len(left_y) / len(parent_y)
    right_weight = len(right_y) / len(parent_y)
    left_gini = gini_index(left_y)
    right_gini = gini_index(right_y)
    information_gain_value = parent_gini - (left_weight * left_gini + right_weight * right_gini)
    return information_gain_value


    


# Objective function to be minimized (-information_gain)
def objective_function(particle):
    return -information_gain(particle, X, y)    


def information_gain_gini_given_particle(X,y,a_particle):
#     print(y.shape)
    weights=a_particle[:-1]
    
    threshold=a_particle[-1]
    products=np.dot(X,weights)
    products=(products-np.min(products))/(np.max(products)-np.min(products))
    X_left=X[np.where(products<threshold)]
    X_right=X[np.where(products>threshold)]
    
#     print(X_left.shape,X_right.shape)
#     print(y.shape,products.shape)

    y_left=y[np.where(products<threshold)]
    y_right=y[np.where(products>threshold)]
    return information_gain_gini(y, y_left, y_right)



class Particle:
    def __init__(self, dim):
        self.position = np.random.rand(dim)  # Initialize random position
        self.velocity = np.random.rand(dim)  # Initialize random velocity
        self.best_position = self.position.copy()  # Initialize personal best position
        self.best_fitness = float('-inf')  # Initialize personal best fitness

        
def apply_PSO(num_particles,num_epochs,X,y,inertia_weight = 0.25, cognitive_weight = 0.5, social_weight = 0.5,mask=None):
    
    if not isinstance(mask,np.ndarray):                
        mask=np.ones(X.shape[1])

    dim=X.shape[1]+1
    swarm=[Particle(dim) for i in range(num_particles)]

    global_best_position = Particle(dim).position
    global_best_fitness = float('-inf')
#     print(global_best_position,global_best_fitness)


    for epoch in range(num_epochs):
    #     print(f"Ep:{epoch}")

        for particle in swarm:
            fitness=information_gain_gini_given_particle(X,y,particle.position)
            if fitness>global_best_fitness:
                global_best_position=particle.position.copy()
                global_best_fitness=fitness
            if fitness>particle.best_fitness:
                particle.best_position=particle.position.copy()
                particle.best_fitness=fitness

    #     print(global_best_fitness)    
        # now update each particle
        for particle in swarm:
            
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            particle.velocity = (inertia_weight * particle.velocity +
                             cognitive_weight * r1 * (particle.best_position - particle.position) +
                             social_weight * r2 * (global_best_position - particle.position))
            particle.position += particle.velocity 
            # here we apply the mask
            # print("Apply mask",particle.position,mask)
            # print("Apply mask",particle.position.shape,mask.shape)
            particle.position[:-1]=particle.position[:-1]*mask
            # print("after masking",particle.position)
            

#     print(f"Finished all epochs. Best particle to be returned with score {global_best_fitness}")
    return global_best_position, global_best_fitness
    
def split(X,y,positions):
    # print("positions",positions)
    weights=positions[:-1]
    threshold=positions[-1]
    # print("X=",X)
    products=np.dot(X,weights)
    # print("After miultip",products,products.shape)
    
    min_products=np.min(products)
    max_products=np.max(products)
    # print("min max products",min_products,max_products)
    products=(products-min_products)/(max_products-min_products)
    # print("Threshold:",threshold)
    # print("products:",products)
    
    X_left=X[np.where(products<threshold)]
    X_right=X[np.where(products>=threshold)]

    y_left=y[np.where(products<threshold)]
    y_right=y[np.where(products>=threshold)]
    
    return X_left,y_left,X_right,y_right, min_products, max_products   


def calculate_leaf_value(Y):
    ''' function to compute leaf node '''
    Y = list(Y)
    return max(Y, key=Y.count)



def build_tree(X,y,curr_depth=0,max_depth=5,max_num_particles=10,max_num_epochs=10,min_split_size=3,mask=None):
    # print("received mask",mask)
    # print("curr",curr_depth,"max_depth",max_depth)
    num_particles=max(max_num_particles*(curr_depth+1),10)
    num_epochs=max(max_num_epochs*(curr_depth+1),10)

    # if X.shape[0]<50:
    #     num_particles=10
    #     num_epochs=10
    
    cntr=Counter(y)
    nm=len(list(cntr.keys()))
    # power=0.75
    # num_epochs=int(max((nm*(X.shape[0]**power))//(curr_depth+1),10))
    # num_particles=int(max(((X.shape[0]*X.shape[1])**power)//(curr_depth+1),10))
    # if num_epochs>100:
    #     num_epochs=100
    # if num_particles>100:
    #     num_particles=100
    # if curr_depth<5:
    #     print("depth",curr_depth)
    # print("curr",curr_depth,"num_particles",num_particles,"num_epochs",num_epochs,"X shape",X.shape,"cntr",cntr)
    
    if curr_depth<max_depth and X.shape[0]>min_split_size and nm!=1:  
        # how about checking if the node is homogeneous
        best_position,best_fitness=apply_PSO(num_particles,num_epochs,X,y,mask=mask)
        # print(f"best_position:{best_position},best_fitness:{best_fitness}")
        X_left,y_left,X_right,y_right,min_products,max_products=split(X,y,best_position)
        
        # print("depth",curr_depth,X.shape,"next",curr_depth+1,X_left.shape,X_right.shape)
        # print("\n",Counter(list(y.reshape(-1))),"\n",Counter(list(y_left.reshape(-1))),"\n",Counter(list(y_right.reshape(-1))))
        if X_left.shape[0]==0 or X_right.shape[0]==0:
#             print("*"*100)
            leaf_value = calculate_leaf_value(y)
            return Node(value=leaf_value)
        # if percentage of any class is more than 95%
        # return Node
        

        left_subtree=build_tree(X_left,y_left,curr_depth=curr_depth+1,max_depth=max_depth,
                                max_num_particles=max_num_particles,max_num_epochs=max_num_epochs,
                                min_split_size=min_split_size,mask=mask)        
        right_subtree=build_tree(X_right,y_right,curr_depth=curr_depth+1,max_depth=max_depth,
                                 max_num_particles=max_num_particles,max_num_epochs=max_num_epochs,
                                 min_split_size=min_split_size,mask=mask)     
        return Node(threshold=best_position[-1],left=left_subtree,right=right_subtree,
                   info_gain=best_fitness,particle=best_position,min_val=min_products,max_val=max_products,
                   samples=X.shape[0])
    
    leaf_value = calculate_leaf_value(y)
    # return leaf node
    return Node(value=leaf_value)        


def make_prediction(x, tree):
    ''' function to predict a single data point '''
    
    if tree.value!=None: return tree.value

    particle=tree.particle
    weights=particle[:-1]
    res=np.dot(x,weights)
#     print(res)

    
    min_val=tree.min_val
    max_val=tree.max_val
    res=(res-min_val)/(max_val-min_val)
    if res<tree.threshold:
        return make_prediction(x,tree.left)
    else:
        return make_prediction(x,tree.right)    
    
    
def predict( X,root):
    ''' function to predict new dataset '''

    preditions = [make_prediction(x,root) for x in X]
    return preditions

def score(predicted_labels, y):
    """ Calculates score accuracy """
    correct = 0
    for i in range(len(y)):
        if predicted_labels[i] == y[i]:
            correct += 1
    return correct / len(y)




## Pruning

# given root node of tree, can we find the average weighted importance of each feature


def traverse_get_weights_samples(root):
    # pre order
    if not root:
        return []
    stack, weights,populations = [(root,0)], [],[]
    
    while stack:
        
        node,level = stack.pop()
        # if node:
        #     print(level,node.samples,node.value)
        if node:
            if isinstance(node.particle,np.ndarray):                
                weights.append(node.particle[:-1])
                populations.append(node.samples)
            stack.append((node.right,level+1))
            stack.append((node.left,level+1))
    print(f"returning {weights}, {populations}")
    return weights,populations


def calculate_weighted_average_particles(weights,populations,prune_rate=1.2):
    total=0
    vals=np.zeros(weights[0].shape[0])
    # print(vals)
    for i in range(len(weights)):
        # print(populations[i],weights[i])
        vals+=np.array(weights[i])*populations[i]
        
        total+=populations[i]
    # print(total)
    # print(vals)
    vals=vals/total
    vals=(vals-np.min(vals))/(np.max(vals)-np.min(vals))
    # print(vals)
    thresh=prune_rate*np.std(vals)
    vals[vals<thresh]=0
    vals[vals>=thresh]=1
    # print(vals)
    return vals


def calculate_feat_importance(weights,populations):
    total=0
    vals=np.zeros(weights[0].shape[0])
    # print(vals)
    for i in range(len(weights)):
        # print(populations[i],weights[i])
        vals+=np.array(weights[i])*populations[i]
        
        total+=populations[i]
    # print(total)
    # print(vals)
    vals=vals/total
    vals=(vals-np.min(vals))/(np.max(vals)-np.min(vals))    
    return vals    
        
def main():
    st.title("PSO-Based Decision Tree Classifier")
    
    # Sidebar options for user input
    st.sidebar.header("Model Parameters")
    depth = st.sidebar.slider("Tree Depth", 6, 10, 6)
    max_num_epochs = st.sidebar.selectbox("Max Num Epochs", [100, 200, 300])
    max_num_particles = st.sidebar.selectbox("Max Num Particles", [5, 10, 20, 50, 100])
    min_split_size = st.sidebar.slider("Min Split Size", 3, 11, 3)
    prune_rate = st.sidebar.slider("Prune Rate", 1.0, 3.5, 1.0, step=0.5)

    # Upload data
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        print("data shape",data.shape)
        print(data.columns)        
        st.write("Data Preview", data.head())
        
        # Assuming the last column is the label
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        X=np.nan_to_num(X)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build and evaluate the initial tree
        tree = build_tree(X_train, y_train, max_depth=depth, max_num_epochs=max_num_epochs*2, max_num_particles=max_num_particles*2, min_split_size=min_split_size)
        y_pred = predict(X_test, tree)
        pre_acc = accuracy_score(y_test, y_pred)
        
        st.write(f"Pre-Pruning Accuracy: {pre_acc:.2f}")
        
        # Pruning
        weights, populations = traverse_get_weights_samples(tree)
        print(weights)


        mask = calculate_weighted_average_particles(weights, populations, prune_rate=prune_rate)
        # st.write(mask)
        compression = len(np.where(mask == 0)[0]) / mask.size
        
        if compression >= 1:
            st.warning(f"Prune rate {prune_rate} is too high. Consider selecting a lower prune rate.")
        else:
            tree_pruned = build_tree(X_train, y_train, max_depth=depth, max_num_epochs=max_num_epochs, max_num_particles=max_num_particles, min_split_size=min_split_size, mask=mask)
            weights_pruned, populations = traverse_get_weights_samples(tree_pruned)
            feat_importance=calculate_feat_importance(weights_pruned,populations)
            feat_importance=feat_importance*mask

            # print(len(data.columns))
            # print(len(feat_importance))
            # print(X.shape)
            



            # Mapping weights to column names
            importance_df = pd.DataFrame({
                'Feature': data.columns[:-1],  # Assuming last column is the label
                'Importance': feat_importance
            })

            print(importance_df.head())

            # Sort by importance
            importance_df = importance_df.sort_values(by='Importance', ascending=False)

            # Plotting the feature importance
            st.write("Feature Importance After Pruning")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
            ax.set_title("Feature Importance")
            st.pyplot(fig)



            y_pred_pruned = predict(X_test, tree_pruned)
            post_acc = accuracy_score(y_test, y_pred_pruned)
            
            st.write(f"Post-Pruning Accuracy: {post_acc:.2f}")
            
            # Precision, Recall, F1 Score
            precision = precision_score(y_test, y_pred_pruned, average="weighted")
            recall = recall_score(y_test, y_pred_pruned, average="weighted")
            f1 = f1_score(y_test, y_pred_pruned, average="weighted")
            
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            st.write(f"F1 Score: {f1:.2f}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred_pruned)
            st.write("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

if __name__ == "__main__":
    main()