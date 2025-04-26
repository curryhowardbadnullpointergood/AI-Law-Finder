package main

import (
	"encoding/csv"
	"fmt"
	"os"
	//"strings"
)

// readCSV reads the CSV file and returns unique expressions
func readCSV(filename string) ([]string, error) {
	// Open the CSV file
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Read all records from the CSV
	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	// Map to store unique expressions
	uniqueExprs := make(map[string]bool)

	// Iterate through each record and add unique expressions to the map
	for _, record := range records {
		if len(record) > 0 {
			expr := record[0]
			uniqueExprs[expr] = true
		}
	}

	// Convert map keys to a slice
	var uniqueExpressions []string
	for expr := range uniqueExprs {
		uniqueExpressions = append(uniqueExpressions, expr)
	}

	return uniqueExpressions, nil
}

func set() {
	// Example: Read and get unique expressions from the CSV
	uniqueExprs, err := readCSV("expressions.csv")
	if err != nil {
		fmt.Println("Error reading CSV:", err)
		return
	}

	// Print the unique expressions
	fmt.Println("Unique expressions:")
	for _, expr := range uniqueExprs {
		fmt.Println(expr)
	}
}
