package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"sync"
)

type OperatorType int

const (
	ADD OperatorType = iota
	SUB
	MUL
	POW
)

type ExprType int

const (
	VAR ExprType = iota
	CONST
	OP
)

type Expr struct {
	Type        ExprType
	VarName     string
	ConstValue  int
	Op          OperatorType
	Left, Right *Expr
}

const (
	maxExpressionsPerDepth = 50000
	numWorkers             = 24
	maxDepth               = 1
)

var operators = []OperatorType{ADD, SUB, MUL, POW}

func createVar(name string) *Expr {
	return &Expr{Type: VAR, VarName: name}
}

func createConst(value int) *Expr {
	return &Expr{Type: CONST, ConstValue: value}
}

func createOp(op OperatorType, left, right *Expr) *Expr {
	return &Expr{
		Type:  OP,
		Op:    op,
		Left:  left,
		Right: right,
	}
}

type job struct {
	a, b *Expr
}

func worker(jobs <-chan job, results chan<- *Expr, wg *sync.WaitGroup) {
	defer wg.Done()

	for pair := range jobs {
		a, b := pair.a, pair.b
		for _, op := range operators {
			expr := createOp(op, a, b)
			results <- expr
		}
	}
}

func printExpression(e *Expr) {
	if e == nil {
		return
	}
	switch e.Type {
	case VAR:
		fmt.Print(e.VarName)
	case CONST:
		fmt.Print(e.ConstValue)
	case OP:
		fmt.Print("(")
		printExpression(e.Left)
		switch e.Op {
		case ADD:
			fmt.Print("+")
		case SUB:
			fmt.Print("-")
		case MUL:
			fmt.Print("*")
		case POW:
			fmt.Print("^")
		}
		printExpression(e.Right)
		fmt.Print(")")
	}
}

func stringifyExpression(e *Expr) string {
	if e == nil {
		return ""
	}
	switch e.Type {
	case VAR:
		return e.VarName
	case CONST:
		return fmt.Sprintf("%d", e.ConstValue)
	case OP:
		left := stringifyExpression(e.Left)
		right := stringifyExpression(e.Right)
		var opSymbol string
		switch e.Op {
		case ADD:
			opSymbol = "+"
		case SUB:
			opSymbol = "-"
		case MUL:
			opSymbol = "*"
		case POW:
			opSymbol = "^"
		}
		return "(" + left + opSymbol + right + ")"
	default:
		return ""
	}
}

func main() {
	symbols := []*Expr{}

	vars := []string{
		"theta1",
		"theta2",
		"theta1_dot",
		"theta2_dot",
		"theta1_ddot",
		"theta2_ddot",
	}

	consts := []int{} // Add constants like g, L1, L2 later

	for _, v := range vars {
		symbols = append(symbols, createVar(v))
	}
	for _, c := range consts {
		symbols = append(symbols, createConst(c))
	}

	expressions := symbols

	for depth := 0; depth < maxDepth; depth++ {
		fmt.Printf("Building depth %d...\n", depth+1)

		jobs := make(chan job, 1000)
		results := make(chan *Expr, 1000)

		var wg sync.WaitGroup

		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			go worker(jobs, results, &wg)
		}

		go func() {
			for i := 0; i < len(expressions); i++ {
				for j := 0; j < len(expressions); j++ {
					jobs <- job{expressions[i], expressions[j]}
				}
			}
			close(jobs)
		}()

		go func() {
			wg.Wait()
			close(results)
		}()

		nextExpressions := []*Expr{}
		for expr := range results {
			nextExpressions = append(nextExpressions, expr)
			if len(nextExpressions) >= maxExpressionsPerDepth {
				break
			}
		}

		expressions = nextExpressions
		fmt.Printf("Depth %d: %d expressions generated.\n", depth+1, len(expressions))
	}

	fmt.Println("\nSample expressions:")
	count := 0
	for _, expr := range expressions {
		if count >= 50 {
			break
		}
		printExpression(expr)
		fmt.Println()
		count++
	}

	file, err := os.Create("expressions.csv")
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	for _, expr := range expressions {
		exprStr := stringifyExpression(expr)
		err := writer.Write([]string{exprStr})
		if err != nil {
			fmt.Println("Error writing record to file:", err)
		}
	}

	fmt.Println("\nSaved expressions to expressions.csv successfully.")

	requiredVars := []string{
		"theta1",
		"theta2",
		"theta1_dot",
		"theta2_dot",
		"theta1_ddot",
		"theta2_ddot",
	}

	filteredExprs, err := readAndFilterExpressions("expressions.csv", requiredVars)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	fmt.Println("Filtered expressions:")
	for i, expr := range filteredExprs {
		if i >= 10 {
			break
		}
		fmt.Println(expr)
	}

	set()

}
