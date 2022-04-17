const express = require('express')


const router = express.Router()

router.get('/', (req, res) => {
    res.render('file.html');
});


module.exports = router
