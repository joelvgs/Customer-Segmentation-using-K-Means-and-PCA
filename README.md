# Customer Segmentation using K-Means and PCA

## Objective
Develop a K-Means Clustering model to segment mall customers based on their annual income and spending behavior, and apply Principal Component Analysis (PCA) to visualize these clusters in two dimensions.

## Dataset Link
[Mall Customer Segmentation Dataset on Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

*(Note: The dataset is not included in this repository to comply with licensing and storage best practices. Please download it from the link above or let the script fetch it locally if placed in the parent directory.)*

## Libraries Used
- **Pandas**: For data loading, manipulation, and preprocessing.
- **NumPy**: For numerical computations.
- **Scikit-Learn**: For preprocessing (`LabelEncoder`, `StandardScaler`), building the clustering model (`KMeans`), and applying dimensionality reduction (`PCA`).
- **Matplotlib & Seaborn**: For creating data visualizations (elbow curve, scatter plots).

## Methodology
1. **Data Understanding**: Loaded the Mall Customers dataset (200 records, 5 features). Identified numerical features (`Age`, `Annual Income (k$)`, `Spending Score (1-100)`) and categorical features (`Gender`).
2. **Data Preprocessing**: Checked for missing values (none found). Removed the unnecessary `CustomerID` column. Encoded `Gender` to numeric values. Standardized all numerical features using `StandardScaler` to ensure features contribute equally to distance calculations in K-Means.
3. **Model Development**:
   - **Elbow Method**: Calculated the Within-Cluster Sum of Squares (WCSS) for $K=1$ to $10$ to find the optimal number of clusters.
   - **K-Means**: Trained the K-Means clustering model using $K=5$ (identified as the optimal number) and assigned cluster labels to each customer.
   - **PCA**: Applied Principal Component Analysis (PCA) to reduce the 4-dimensional dataset into 2 principal components (retaining ~59.92% of the total variance) for 2D visualization.
4. **Visualization and Evaluation**: Generated the Elbow Curve to validate $K=5$, a standard scatter plot of Income vs Spending Score colored by cluster, and a PCA-transformed scatter plot showcasing the 5 clusters in the reduced 2D space.

## Results
- **Optimal Clusters ($K$)**: 5
- **PCA Variance Explained**: ~59.92% with 2 components.
- **Cluster Characteristics**: The K-Means model successfully identified 5 distinct customer groups which are most visibly separated by Annual Income and Spending Score:
  1. High Income / High Spend
  2. High Income / Low Spend
  3. Average Income / Average Spend
  4. Low Income / High Spend
  5. Low Income / Low Spend

## Conclusion
Through K-Means clustering, we successfully segmented the mall customers into 5 distinct groups based on their age, gender, income, and spending score. Visualizing the data using Principal Component Analysis (PCA) clearly revealed these separated clusters in 2D space. 

Business Applications: Identifying these segments allows the mall to deploy targeted marketing campaigns. For example, customers in the "High Income/High Spend" cluster can be targeted for luxury products, while "High Income/Low Spend" customers can be targeted with promotional discounts to encourage more spending.

Limitation of K-Means: One limitation of K-Means clustering is that it requires the number of clusters (K) to be specified in advance, and it assumes spherical clusters, which may not always fit real-world, complex data shapes.

Advantage of PCA: One major advantage of PCA is dimensionality reduction. It reduces the number of variables while preserving as much variance (information) as possible, significantly improving visualization and speeding up machine learning algorithms without a severe loss of data fidelity.
