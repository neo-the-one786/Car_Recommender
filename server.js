import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import csv from 'csv-parser';
import {createServer} from 'http';
import {fileURLToPath} from 'url';
import {dirname} from 'path';
import natural from 'natural';

// Get the directory name for module path
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const port = 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({extended: true}));

// Load and preprocess data
const data = [];
fs.createReadStream(path.join(__dirname, 'train.csv'))
    .pipe(csv())
    .on('data', (row) => {
        row.Mileage = parseFloat(row.Mileage.replace(' km', '').replace(',', ''));
        row.Levy = row.Levy === '-' ? null : parseFloat(row.Levy);
        row.ID = row.ID.trim();  // Ensure no extra spaces
        data.push(row);
    })
    .on('end', () => {
        console.log('CSV file successfully processed');
        console.log('Data Length:', data.length);  // Debugging
    });

const tokenizer = new natural.WordTokenizer();
const tfidf = new natural.TfIdf();

app.get('/recommend', (req, res) => {
    const carId = parseInt(req.query.car_id, 10);
    if (isNaN(carId)) {
        return res.status(400).json({error: 'Invalid Car ID'});
    }

    console.log('Requested Car ID:', carId);  // Debugging

    // Check all IDs in data
    const allIds = data.map(item => parseInt(item.ID, 10));
    console.log('All IDs:', allIds);  // Debugging

    const car = data.find(item => Number(item.ID) === carId);
    if (!car) {
        return res.status(404).json({error: 'Car ID not found'});
    }

    // Combine features into a single string for TF-IDF
    const combineFeatures = (row) => [
        row.Manufacturer,
        row.Model,
        row.Category,
        row['Leather interior'],
        row['Fuel type'],
        row['Gear box type'],
        row['Drive wheels'],
        row.Color,
        row['Engine volume'],
        row.Mileage,
        row.Price
    ].join(' ');

    // Tokenize and add documents to TF-IDF instance
    data.forEach(item => {
        const text = combineFeatures(item);
        tfidf.addDocument(tokenizer.tokenize(text).join(' '));
    });

    // Tokenize and query TF-IDF instance
    const queryText = combineFeatures(car);
    const queryTokens = tokenizer.tokenize(queryText).join(' ');

    // Compute similarities
    const similarities = tfidf.listTerms(0).map((term, index) => ({
        index,
        similarity: tfidf.tfidf(queryTokens, index)
    }));

    // Sort and select top 10 similar cars
    similarities.sort((a, b) => b.similarity - a.similarity);
    const similarCars = similarities
        .filter(item => item.index !== data.findIndex(item => Number(item.ID) === carId))
        .slice(0, 10)
        .map(({index}) => data[index]);

    res.json(similarCars.map(({ID, Manufacturer, Model, Price, Mileage}) => ({
        ID,
        Manufacturer,
        Model,
        Price,
        Mileage
    })));
});

createServer(app).listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
