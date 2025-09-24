SmartHarvest - Smart Crop Prediction 

Overview
SmartHarvest is a web-based demonstration application designed to provide crop recommendations to farmers. The tool, titled "SmartHarvest - Smart Crop Prediction," aims to help users "Leverage predictive analytics to choose the most profitable crops for your land and season". Users can select a geographical region and a planting month to receive a prediction for the most suitable and profitable crop.

This application is a client-side prototype and uses a mock dataset for its predictions. It serves as a proof-of-concept for a smart farming tool.

Features
Recommendation Engine: Predicts the best crop based on user inputs for the planting season (month) and region.

Region-Specific Data: Includes options for five distinct agricultural zones: Northern Plains, Southern Plateau, Coastal Areas, Himalayan Region, and Desert Region.

Detailed Prediction Results: The output provides comprehensive details for the recommended crop, including:

A descriptive summary.

Predicted Profitability (e.g., High, Very High).

Confidence Level percentage.

Predicted Yield in Quintals/Acre.

Predicted Profit in ₹/Acre.

A list of potential commercial products from the crop (e.g., Flour, Mustard Oil, Textiles).

Yield Trend Visualization: A dynamic bar chart displays the "Historical & Predicted Yield Trend," showing data for two past years ("Actual") and three future years ("Predicted").

User-Friendly Interface: A clean, responsive interface with a loading animation to simulate data analysis.

How It Works
The SmartHarvest application operates entirely on the client-side within the browser.

Mock Data Source: All crop predictions are sourced from a hardcoded JavaScript object named cropData within the index1.html file. This object contains pre-defined crop recommendations for each region and month combination, along with associated metrics like profitability, yield, and potential products.

Prediction Simulation: When a user clicks the "Predict Best Crop" button, the application performs a lookup in the cropData object based on the selected inputs. To enhance the user experience and mimic a real analytical process, a 1.5-second delay is intentionally added using setTimeout before displaying the results.

Technical Stack
The application is built using standard web technologies and relies on CDN-hosted libraries:

Structure: HTML

Logic: JavaScript

Getting Started:
No complex setup or local server is required to run this application.

Open the file directly in any modern web browser (e.g., Google Chrome, Firefox, Microsoft Edge).

And can be viewed on: https://g.co/gemini/share/82ccdeff8b98

Contributors:
Manan Singhal 
Viransh Jain 
Smit Shubhanshu Kamatnurkar
Animesh Panda 
Frontend: User interface (HTML), styling (CSS), and client-side logic (JavaScript).

Backend: This application is currently a client-side prototype and does not have a dedicated backend. A backend developer would be responsible for creating an API, managing a database, and implementing a real prediction model.
