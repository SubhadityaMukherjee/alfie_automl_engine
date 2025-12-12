# AutoDW Setup

- Clone the repository : https://github.com/sstamatis01/ALFIE-Data-Warehouse
- cd into it
- Do a `docker-compose up -d`
- Open the swagger UI : `http://localhost:8000/docs#/`
    - Upload a dataset : POST [/dataset/upload/{user_id}] (Click try it out and fill in the details)
    - Note the user id and dataset id of course, then you can use it with the AutoML tool
- !Note: All this is only temporary, once the actual service is up and running, there will be a proper UI for everything

