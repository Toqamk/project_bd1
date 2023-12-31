# Use Ubuntu as the base image
FROM ubuntu

# Set the working directory inside the container
WORKDIR /home/bd-a1

# Install required packages
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install pandas numpy seaborn matplotlib scikit-learn scipy

# Install Jupyter Notebook
#RUN pip3 install jupyter


# Create a directory inside the container
RUN mkdir /home/bd-a1/service-result/
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim"]
#

# Copy the dataset to the container
COPY ford.csv /home/bd-a1/
COPY load.py /home/bd-a1/
COPY preprocess.py /home/bd-a1/
COPY visualization.py /home/bd-a1/
COPY knn.py /home/bd-a1/
COPY decisiontree.py /home/bd-a1/
COPY gradient.py /home/bd-a1/
COPY svr.py /home/bd-a1/
COPY randomforest.py /home/bd-a1/
COPY linear.py /home/bd-a1/

# Open the bash shell upon container startup
CMD ["/bin/bash"]
