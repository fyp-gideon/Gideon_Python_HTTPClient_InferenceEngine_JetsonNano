// The require('debug') returns a function here.
// We invoke this function by providing an argument which is our intended namespace for debugging.
const express = require ('express');
const app = express();

// Built-in Middle-ware to serve static assets/files/content such as images, css etc., serving from the root of the website:
app.use(express.static('public'));

// Setting up PORT environment variable.
// Using 'process' object's 'env' property and then it's public member PORT.
// In terminal > set PORT=5000
const port = process.env.PORT || 3000;
// Step #2:
// Listening on a given port. An optional callback function can be provided that would be invoked after starting server on the mentioned port.
app.listen(port, () => console.log(`Server started. Listening on Localhost Port ${port} ... `));