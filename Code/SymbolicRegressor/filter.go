package main

import (
	"encoding/csv"
	//"fmt"
	"os"
	"strings"
)

// readAndFilterExpressions reads expressions from a CSV and filters them
func readAndFilterExpressions(filename string, requiredVars []string) ([]string, error) {
	// Open the file
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

	filtered := []string{}

	for _, record := range records {
		if len(record) == 0 {
			continue
		}
		expr := record[0]

		matchesAll := true
		for _, v := range requiredVars {
			if !strings.Contains(expr, v) {
				matchesAll = false
				break
			}
		}

		if matchesAll {
			filtered = append(filtered, expr)
		}
	}

	return filtered, nil
}
