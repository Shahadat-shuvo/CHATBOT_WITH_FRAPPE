## Mybot

CHAT WITH PDF
A chatbot using the Frappe framework that can read a given PDF, and answer questions accordingly. There are three ways so far to build this, by creating doctype, by creating custom pages and by creating Custom portal pages. I have built this using Portal pages. The reason behind I choose this because there is no need to define route, the route will automatically generated by html file name, no need to set hook for js and python script just set these with the same name as the html file. Throughout to build this site I have also introduced the Large Language framework LANGCHAIN. 

Structure:      
![Basic Structure](.C:\Users\Shuvo\Pictures\Screenshots\Screenshot (390).png)




Details:
frappe.call() method is the powerful method that can interact with server side from client side script. There is another way like frm.call() method but frappe.call() is more convenient to handle response. (apps/mybot/mybot/www/mybots.js)
Here in the code snippet shows with a frappe call method, the method call a python function named get_chat_response with the user question as user input, the function returns a response to the callback function to handle it. ((apps/mybot/mybot/www/mybots.py))			

I have add another frappe.call() method to retrieve chat history and display it with the same process mentioned above-
There is also a memory to follow up the previous question. 


#### License

MIT
