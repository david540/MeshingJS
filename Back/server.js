const chalk = require('chalk');
const express = require('express')
const engines = require('consolidate');
const compression = require('compression');
const logger = require('./logger.js');
const port = 8080
const server = express()
const bodyParser = require('body-parser');


////////////////////// setup /////////////////////////////////
server.use(express.static("../Front/"));
server.use('/', require("./router"))

server.set('views', "../Front");
server.engine('html', engines.mustache);
server.set('view engine', 'html');


////////////////////// run /////////////////////////////////
server.listen(port, '0.0.0.0', () => {
  console.log("Starting up http-server, serving...")
  console.log("Available on:")
  console.log(`${chalk.green("Local:")} http://localhost:8080"`)
})
