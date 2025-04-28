package main

import (
	"encoding/csv"
	"fmt"
	"os"
)

func readCSV(filename string) ([]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	uniqueExprs := make(map[string]bool)

	for _, record := range records {
		if len(record) > 0 {
			expr := record[0]
			uniqueExprs[expr] = true
		}
	}

	var uniqueExpressions []string
	for expr := range uniqueExprs {
		uniqueExpressions = append(uniqueExpressions, expr)
	}

	return uniqueExpressions, nil
}

func set() {
	uniqueExprs, err := readCSV("expressions.csv")
	if err != nil {
		fmt.Println("Error reading CSV:", err)
		return
	}

	fmt.Println("Unique expressions:")
	for _, expr := range uniqueExprs {
		fmt.Println(expr)
	}
}
