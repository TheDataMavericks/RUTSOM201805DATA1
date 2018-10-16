## 13.1 Lesson Plan - Mastering MongoDB

### Please take the Mid-Course Instructional Staff Survey if You Haven't Yet

Trilogy, as a company, values transparency and data-driven change quite highly. As we grow, we know there will be areas that need improvement. It’s hard for us to know what these areas are unless we’re asking questions. Your candid input truly matters to us, as you are key members of the Trilogy team. In addition to the individual feedback at the end of lesson plans
we would appreciate your feedback at the following link if you have not already taken the mid-course survey:
[https://docs.google.com/forms/d/e/1FAIpQLSdWXdBydy047_Ys1wm6D5iJY_J-0Mo0BqCjfGc4Er2Bz9jo5g/viewform](https://docs.google.com/forms/d/e/1FAIpQLSdWXdBydy047_Ys1wm6D5iJY_J-0Mo0BqCjfGc4Er2Bz9jo5g/viewform)

### Overview

In this class, students will be introduced to the concept of the NoSQL database with MongoDB. By the end of the day, students should be able to perform basic CRUD operations with MongoDB and the Pymongo library.

### Instructor Priorities

* Students should be able to install MongoDB on their machines within the first hour of class. If anyone has trouble getting it running, the instructional team should strive to offer that student assistance.

* Students should understand how to make queries with MongoDB by the end of class. Meeting this goal will build the necessary foundation for the next lecture which will integrate such queries with web-scraping.

### Sample Class Video (Highly Recommended)

* To view an example class lecture visit (Note video may not reflect latest lesson plan): [Class Video](https://codingbootcamp.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=c1d0ef7e-7378-46b7-96dd-664c87a0163c)

- - -

### Class Objectives

* Students will be able to create and connect to local MongoDB databases

* Students will learn to create, read, update, and delete MongoDB documents using the Mongo Shell

* Students will create simple Python applications that connect to and modify MongoDB databases using the PyMongo library

- - -

### 01. Instructor Do: What is MongoDB (Slides) (0:10)

* Open the [Masters of MongoDB](MastersOfMongoDB.pptx) slide deck and go through the presentation with the class, answering whatever questions students may have.

* Start by informing the class that MongoDB is a popular NoSQL database.

  * A NoSQL database is simply a non-relational database. In other words, NoSQL databases do not employ SQL relational model when storing data.

  * Students should recall their experiences working with JSON in the past. MongoDB uses a very similar format called BSON which stands for Binary JSON.

  * For the sake of simplicity and the purposes of this class, let students know that working with BSON documents is essentially identical to working with JSON.

* The key differences between SQL and NoSQL databases can be seen in how related data points are stored in each.

  * In SQL databases, relating data between tables requires the developer to join the rows of one with the rows of another.

    ![SQL Joins](Images/01-WhatIsMongo_SQLJoins.jpg)

  * BSON data, on the other hand, do not require much in the way of joins because they can store objects within objects. This allows developers to save nested data directly and eliminates the need to model data relationally.

  * With NoSQL, once data is added to the database, it is a cinch to traverse. Simply navigate through the data in the same manner one would JSON data.

    ![NoSQL Data Storagesss](Images/01-WhatIsMongo_NoSQL.jpg)

* Open the MongoDB storage slide and explain to the class how MongoDB databases store data.

  * Databases contain collections. Collections contain documents. Documents contain fields. Fields store data.

    ![Collection](Images/01-WhatIsMongo_Collection.jpg)

  * Reiterate how MongoDB is still an inherently different style of data storage than MySQL. A BSON document is basically a more flexible form of JSON with individual documents capable of containing strings, ints, booleans, arrays, and even other objects.

* Answer whatever questions students may have before moving onto the next activity.

### 02. Students Do: Quick Mongo Research (Slides) (0:05)

* Load up the next slide with the instructions for this activity and tell the class to work with a partner to answer the following questions:

  * What are the advantages of using a NoSQL database like MongoDB according to the MongoDB Website?

  * What are the advantages of using a NoSQL database like MongoDB according to the web (places like Quora)?

  * What are the disadvantages of using a NoSQL database like MongoDB according to the web (places like Quora)?

### 03. Everyone Do: Quick Mongo Research Review (Slides) (0:05)

* Ensure students can see the slide with the activity questions whilst reviewing the previous activity with the class..

* What are the advantages of using a noSQL database like MongoDB according to the MongoDB Website?

  * "Relational databases require that schemas be defined before you can add data. For example, you might want to store data about your customers such as phone numbers, first and last name, address, city and state – a SQL database needs to know what you are storing in advance."

  * "Object-oriented programming that is easy to use and flexible."

* What are the advantages of using a noSQL database like MongoDB according to the web? [Question Link](http://stackoverflow.com/questions/2117372/what-are-the-advantages-of-using-a-schema-free-database-like-mongodb-compared-to)

  * Deep query-ability. MongoDB supports dynamic queries on documents using a document-based query language that's nearly as powerful as SQL.

  * No schema migrations. Since MongoDB is schema-free, your code defines your schema.

* What are the disadvantages of using a NoSQL database like MongoDB according to the web? [Question Link](http://stackoverflow.com/questions/2117372/what-are-the-advantages-of-using-a-schema-free-database-like-mongodb-compared-to)

  * Sometimes, using joins and having strict schemas is actually preferable to MongoDB.

  * "If your database has a lot of relations and normalization, it might make little sense to use something like MongoDB. It's all about finding the right tool for the job."

### 04. Students Do: Installing MongoDB (0:20)

* Tell students to consult the [installation guide](Supplemental/Installing-MongoDB.md) and to take 15 to 20 minutes to install and configure MongoDB on their machines.

  * Tell them to ask the instructional team for help if they have any questions while installing or configuring MongoDB.

* At the 15-minute mark, ask if there are any people who haven't been able to install and configure MongoDB yet. Assist anyone who needs help.

* Ask the class if they can start up MongoDB by typing `mongod` into their terminal/bash windows. Their terminal/bash screens should look something like this:

    ![5-mongod](Images/02-Install_Mongod.jpg)

* If there are any remaining students who do not have it installed and configured, ask them to talk with a TA to figure out the issue.

### 05. Instructor Do: Basic MongoDB Queries (0:15)

* Instruct the class to open `mongod` if they don't already have it open and to follow along throughout this activity.

  * The `mongod` window must remain open so that MongoDB can continue to run.

  * While `mongod` is running, open up another terminal/bash window and run `mongo` to start up the mongo shell.

* As with MySQL, the first step in working with any kind of database is to create that database on the server.

  * Create the database `travel_db` by typing the command `use travel_db` into the mongo shell.

  * The existence of this database can be verified using the `db` command. This command lets users know which database they are currently working inside of.

  * To show all of the databases that currently exist on the server, type `show dbs` into the Mongo shell.

    ![Databases](Images/03-BasicMongo_DB.png)

  * Only those databases that contain some data will be shown. MongoDB will not save a database until some values exist within it.

* To show the collections within the current database, enter `show collections` into the mongo shell.

  * Because no collection has been created within `travel_db` yet, nothing will be returned at this time.

  * To insert a document into a database's collection, the syntax `db.collectionName.insert({key:value})` is used.

    ![Inserting Data](Images/03-BasicMongo_Collections.png)

  * The `db` implicitly refers to the currently selected database. In this specific case that means it is referring to `travel_db`.

  * `collectionName` should be replaced with the name of the collection that the data will be inserted into. If the named collection does not yet exist, then mongo will create it automatically.

  * `insert({key:value})` allows users to then insert a document into the collection. Remind the students that the format of the document in functionally similar to that of a Python dictionary.

  * `db.collectionName.find().pretty()` can then be used in order to print out the data that are stored within the named collection. The `pretty()` method prints out the data in a more readable format.

    ![Pretty Print](Images/03-BasicMongo_PrettyPrint.png)

  * With the assistance of the class, insert two or three new documents into the `destinations` collection before moving onto the next point.

* To find specific documents within a collection, the syntax used is `db.collectionName.find({key:value})`.

    ![Finding Documents](Images/03-BasicMongo_Find.png)

  * The first line of code above will perform a search for all documents whose `country` matches `USA`.

  * The second line will return all documents whose `continent` is `Europe`.

  * Lastly, explain how it is possible to find a single document by its `_id` which is a uniquely generated string that is automatically set whenever a document is made.

* Answer any questions students may have before moving onto the next activity.

### 06. Students Do: Mongo Class (0:15)

* In this activity, students will familiarize themselves with the basic query operations in MongoDB. Specifically, they will practice inserting and finding documents.

  ![Mongo Class Output](Images/04-MongoClass_Output.png)

* **Instructions**:

  * Use the command line to create a `ClassDB` database

  * Insert entries into this database for yourself and the people around you within a collection called `students`

  * Each document should have a field of `name` with the person's name, a field of `favoriteLibrary` for the person's favorite Python library, a field of `age` for the person's age, and a field of `hobbies` which will hold a list of that person's hobbies.

  * Use the `find()` commands to get a list of everyone of a specific age before using `name` to collect the entry for a single person.

* **Bonus**:

  * Check out the MongoDB documentation and figure out how to find users by an entry within an array.

### 07. Everyone Do: Mongo Class Review (0:05)

* Open up [02-Stu_MongoClass](Activities/02-Stu_MongoClass/Solved/MongoClass.md) within an editor and go over the code contained within with the class, answering whatever questions students may have.

* When discussing this activity, make sure to hit upon the following points...

  * Creating/selecting a database is simple: `use classDB`, where `classDB` is the name of the database.

  * Inserting a document into a collection is also simple. The syntax involved is: `db.students.insert({})`, where `students` is the name of the collection, and a document in the form of a dictionary is inserted between the parentheses.

    ![Inserting Documents](Images/04-MongoClass_Insert.png)

### 08. Instructor Do: Removing, Updating and Dropping in MongoDB (0:10)

* Now that the class knows how to create and read elements within a Mongo database, it is time to go over how to update and delete data as well.

* Using the `travel_db` database from earlier, show the class how they can use the `db.collectionName.update()` method to update documents.

  * The `update()` method takes in two objects as its parameters. The first object tells the application what document(s) to search through whilst the second object informs the application on what values to change.

  * The second object passed uses the following syntax: `{$set: {KEY:VALUE}}`. Failing to use this syntax could lead to errors or might even break the document.

  * Make sure to let the class know that the `update()` method will only update the first entry which matches.

  * To update more than one document, the `updateMany()` method can be used instead. This method will update all of the records that meet the given criterion.

  * Passing the parameter `{multi:true}` into the `update()` method would also work but is more complex and not ideal.

    ![Updating Mongo](Images/05-MongoCRUD_Update.png)

  * If the field to update does not yet exist, the field will be inserted into the document instead.

  * If the document being searched for within a collection does not exist, the `update()` method will not create the document in question unless `{upsert:true}` is passed as a parameter. This option combines `update` and `insert`, meaning that if a document already exists that meets the given criterion, it will be updated. If the document doesn't exist, however, MongoDB will create one with the given information.

    ![Upsert Option](Images/05-MongoCRUD_Upsert.png)

  * To add elements into an array, use `$push` instead of `$set`. This will push the value provided into the array without modifying any of the other elements.

    ![Push to Array](Images/05-MongoCRUD_Push.png)

* Deleting documents from a Mongo collection is easy as the `db.collectionName.remove({})` method is used.

  * The object being passed into the `remove()` method dictates what key/value pairing to search for. Adding the `justOne` parameter will remove a single document.

  * Without the `justOne` parameter, all documents matching the key/value pairing will be dropped from the collection.

  * Passing an empty object into the `remove()` method will drop all documents from the collection. This is extremely risky as all of that data will be lost.

  * The `db.collectionName.drop()` method will delete the collection named from the Mongo database while `db.dropDatabase()` will delete the database itself.

    ```python
    # Show how to delete an entry with db.[COLLECTION_NAME].remove({justOne: true})
    db.destinations.remove({"country": "Morocco"},
    {justOne: true})

    # Show how to empty a collection with db.[COLLECTION_NAME].remove()
    db.destinations.remove({})

    # Show how to drop a collection with db.[COLLECTION_NAME].drop()
    db.destinations.drop()

    # Show how to drop a database
    db.dropDatabase()
    ```

* Answer any questions the class may have before moving on to the next activity.

### 09. Students Do: Dumpster_DB (0:15)

* In this activity, students will gain further practice with CRUD operations in MongoDB as they create a database centered around dumpster diving.

  ![Dumpster DB Output](Images/06-DumpsterDB_Output.png)

* **Instructions**:

  * Create and use a new database called `Dumpster_DB` using the Mongo shell.

  * Create a collection called `divers` which will contain a string field for `name`, an integer field for `yearsDiving`, a boolean field for `stillDiving`, and an array of strings for `bestFinds`.

  * Insert three new documents into the collection. Be creative with what you put in here and have some fun with it.

  * Update the `yearsDiving` fields for your documents so that they are one greater than their original values.

  * Update the `stillDiving` value for one of the documents so that it is now false.

  * Push a new value into the `bestFinds` array for one of the documents.

  * Look through the collection, find the diver with the smallest number of `bestFinds`, and remove it from the collection.

### 10. Everyone Do: Dumpster_DB Review (0:05)

* Open up [04-Stu_DumpsterDB](Activities/04-Stu_DumpsterDB/Solved/dumpsterDB.md) within an editor and go over the code contained within with the class, answering whatever questions students may have.

### 11. Instructor Do: Mongo Compass (0:10)

* While working within the mongo shell is fine and dandy, life would be far simpler if there were an application to view/modify Mongo databases. Thankfully there is in [MongoDB Compass](https://www.mongodb.com/products/compass).

  * Students may have already installed MongoDB Compass during their installation of MongoDB Server. If so, they should be able to open up the application already. If not, have them download the application from [this link](https://www.mongodb.com/download-center?filter=enterprise#compass).

* Once the class has downloaded and installed MongoDB Compass, open up the application and walk through how to connect to localhost with the students.

  * Connecting to localhost is REALLY simple with MongoDB Compass as the default values for the connection are always set for localhost. Because of this, the class should be able to connect straight away so long as `mongod` is running.

    ![Mongo Compass Connect](Images/07-MongoCompass_Connect.png)

  * Upon hitting that "CONNECT" button, students should be able to view a list of all of the MongoDB databases hosted on their localhost server.

  * Clicking on a database's name will take users to a list of all of the collections stored on that database. Clicking on a collection name will then take them into a view in which they can peruse all of that collection's documents.

    ![Compass Docs View](Images/07-MongoCompass_DocsView.png)

* When inside of the Document Viewer, users can create, read, update, and even delete data using the GUI. They can also choose to view their data as a table if they really wanted to.

### 12. Students Do: Compass Playground (0:05)

* Now that the class has MongoDB Compass installed on their computers, provide them with some time to play around with the application.

* After time has passed, release the class on their break and let them know that they will be diving back into Python when they return.

- - -

### 13. BREAK (0:15)

- - -

### 14. Instructor Do: Introduction to Pymongo (0:15)

* This activity introduces the use of the Pymongo library which allows developers to use Python to work with MongoDB.

  * Inform students that Pymongo serves as the interface between Python and MongoDB.

  * The syntax used in Pymongo is strikingly similar to that of MongoDB. As such, the learning curve for the library is quite small in comparisson to something like SQLAlchemy.

* Open up [05-Ins_PyMongo](Activities/05-Ins_PyMongo/Solved/IntroToPymongo.py) within an IDE and work through the code line-by-line with the class, answering whatever questions students may have.

  * After importing the PyMongo library into the application, a connection with a running instance of MongoDB must be established using `pymongo.MongoClient(connectionString)`

  * As was the case with SQLAlchemy, the connection PyMongo establishes is set with a connection string. This string uses the syntax `mongodb://USERNAME:PASSWORD@HOST:PORT`

  * Since the default localhost connection does not have a username or password set, the string for local instances of MongoDB would be `mongodb://localhost:27017`

  * Explain that `27017` is the default port used by MongoDB. It also happens to be a zip code in South Carolina.

    ![PyMongo Connection](Images/08-PyMongo_Connection.png)

* The `classDB` database is assigned to the variable `db` using `client.classDB`. This tells the PyMongo client that the developer will be working inside of the `classDB` database.

  * The `db.collectionName.find({})` method creates a query that collects all of the documents within the collection named.

  * The query can be made more specific by adding key/value pairs into the object passed as a parameter.

  * Inserting a document into a collection in Pymongo is similar to the process in MongoDB. Here, the only difference is the underscore used in the `insert_one()` method, in contrast to the camel case used in MongoDB's `insertOne()`.

    ![Read and Create](Images/08-PyMongo_ReadCreate.png)

  * Likewise, updating a document in Pymongo is similar to its counterpart in MongoDB. Again, the only difference is the underscore used in `update_one()`.

  * Remind the class that after specifying the field with which we identify the document to be updated, the information to be updated is specified with the syntax: `{$set: {key:value}}`.

  * Pushing an item into an array is similar with `$set` getting replaced with `$push` instead.

    ![PyMongo Update](Images/08-PyMongo_Update.png)

  * To delete a field from a document, the `update_one({},{})` method can be used and `$unset` is passed into the second object in place of `$set`.

  * Finally, go over how to delete a document from a collection using the `db.collectionName.delete_one({})` method where the document to delete has data matching that stored within the passed object.

    ![PyMongo Delete](Images/08-PyMongo_Delete.png)

* Answer any questions students may have before moving on to the next activity.

### 15. Students Do: Mongo Grove (0:25)

* In this activity, students will build a command-line interface application for the produce department of a supermarket. They will have to use PyMongo to enable Python to interact with MongoDB.

* **Instructions**:

  * Use Pymongo to create a `fruits_db` database, and a `fruits` collection.

  * Into that collection, insert two documents of fruit shipments received by your supermarket. They should contain the following information: vendor name, type of fruit, quantity received, and ripeness rating (1 for unripe, 2 for ripe, 3 for over-ripe).

  * Because not every supermarket employee is versed in using MongoDB, your task is to build an easy-to-use app that can be run from the console.

  * Build a Python script that asks the user for the above information, then inserts a document into a MongoDB database.

  * It would be good to Modify the app so that when the record is entered, the current date and time is automatically inserted into the document.

* **Hint**:

  * Consult the [documentation](https://docs.python.org/3/library/datetime.html) on the `datetime` library.

### 16. Everyone Do: Mongo Grove Review (0:05)

* Open up [06-Stu_MongGrove](Activities/06-Stu_MongGrove/Solved/mongo_grove.py) within an IDE and go over the code contained within with the class, answering whatever questions students may have.

  * A connection string is created and set to the variable `conn` before being used to create a connection to a local MongoDB server.

  * After declaring the database and the collection as `db` and `collection`, the user's input is used to set the `vendor`, `fruit_type`, `quantity`, and `ripeness` variables. These items are then inserted as key-value pairs within a dictionary.

  * Point out that `datetime.datetime.utcnow()` can be used as the value of a key-value pair to be inserted as a timestamp of the data entry.

  * In order to insert the dictionary created as a new document, the `insert_one()` method is used.

  * To print the current inventory within the collection, a `find()` query is used on `fruits_db` and then the results are looped through.

- - -

### LessonPlan & Slideshow Instructor Feedback

* Please click the link which best represents your overall feeling regarding today's class. It will link you to a form which allows you to submit additional (optional) feedback.

* [:heart_eyes: Great](https://www.surveygizmo.com/s3/4381674/DataViz-Instructor-Feedback?section=13.1&lp_useful=great)

* [:grinning: Like](https://www.surveygizmo.com/s3/4381674/DataViz-Instructor-Feedback?section=13.1&lp_useful=like)

* [:neutral_face: Neutral](https://www.surveygizmo.com/s3/4381674/DataViz-Instructor-Feedback?section=13.1&lp_useful=neutral)

* [:confounded: Dislike](https://www.surveygizmo.com/s3/4381674/DataViz-Instructor-Feedback?section=13.1&lp_useful=dislike)

* [:triumph: Not Great](https://www.surveygizmo.com/s3/4381674/DataViz-Instructor-Feedback?section=13.1&lp_useful=not%great)

- - -

### Copyright

Trilogy Education Services © 2017. All Rights Reserved.
