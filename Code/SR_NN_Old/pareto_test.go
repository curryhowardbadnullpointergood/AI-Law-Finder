package main

import (
	// "github.com/stretchr/testify/assert"
	"log"
	"os"
	"reflect"
	"testing"
)

func TestGetIdList(t *testing.T) {
	ps := NewParetoSet()

	// Add sample points
	ps.points = []Point{
		{Id: 101, x: 1.0, y: 2.0},
		{Id: 102, x: 2.0, y: 1.5},
		{Id: 103, x: 0.5, y: 3.0},
	}

	expected := []int{101, 102, 103}
	result := ps.GetIdList()

	if !reflect.DeepEqual(expected, result) {
		t.Errorf("GetIdList() = %v; want %v", result, expected)
	}

}

func TestToXYArray(t *testing.T) {
	ps := NewParetoSet()

	ps.points = []Point{
		{x: 1.0, y: 2.0},
		{x: 3.5, y: 0.5},
		{x: 2.2, y: 1.1},
	}

	expected := [][]float64{
		{1.0, 2.0},
		{3.5, 0.5},
		{2.2, 1.1},
	}

	result := ps.ToXYArray()

	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ToXYArray() = %v; want %v", result, expected)
	}
}

func TestGetParetoPoints(t *testing.T) {
	ps := NewParetoSet()

	ps.points = []Point{
		{x: 1.0, y: 2.0, Data: "A"},
		{x: 3.5, y: 0.5, Data: "B"},
		{x: 2.2, y: 1.1, Data: 42},
	}

	expected := [][]interface{}{
		{1.0, 2.0, "A"},
		{3.5, 0.5, "B"},
		{2.2, 1.1, 42},
	}

	result := ps.GetParetoPoints()

	if !reflect.DeepEqual(result, expected) {
		t.Errorf("GetParetoPoints() = %v; want %v", result, expected)
	}
}

func TestPlot(t *testing.T) {
	ps := NewParetoSet()

	ps.points = []Point{
		{x: 1.0, y: 2.0},
		{x: 3.0, y: 1.0},
		{x: 2.0, y: 1.5},
	}

	filename := "/home/fish/Documents/University/Part3_Project/Part3/Code/SR_NN/test.png"

	// Clean up if file exists
	_ = os.Remove(filename)

	ps.Plot(filename)

	// Optional: check if the file exists
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		t.Errorf("plot file was not created")
	} else {
		t.Logf("Plot saved at: %s", filename)
	}

	// Clean up after test
	_ = os.Remove(filename)
}

func TestLoggingToFile(t *testing.T) {
	// Create or open the log file
	logFile, err := os.OpenFile("test_output.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		t.Fatalf("Failed to open log file: %v", err)
	}
	defer logFile.Close()

	// Set log output to the file
	logger := log.New(logFile, "TEST_LOG: ", log.Ldate|log.Ltime)

	// Example logs
	logger.Println("Running TestLoggingToFile")

	// You can still use t.Log or t.Error too
	t.Log("TestLoggingToFile ran successfully")
}

func TestContainsPoint(t *testing.T) {
	// Create some points to test with
	point1 := &Point{x: 1.0, y: 2.0, Id: 1}
	point2 := &Point{x: 3.0, y: 4.0, Id: 2}
	point3 := &Point{x: 5.0, y: 6.0, Id: 3}
	pointNotInSet := &Point{x: 7.0, y: 8.0, Id: 4}

	// Create a ParetoSet
	ps := &ParetoSet{
		points: []Point{
			{x: 1.0, y: 2.0, Id: 1},
			{x: 3.0, y: 4.0, Id: 2},
			{x: 5.0, y: 6.0, Id: 3},
		},
	}

	// Test 1: The point should be in the set (same x and y)
	if !ps.ContainsPoint(point1) {
		t.Errorf("Expected point %v to be in the ParetoSet", point1)
	}

	// Test 2: The point should be in the set (same x and y)
	if !ps.ContainsPoint(point2) {
		t.Errorf("Expected point %v to be in the ParetoSet", point2)
	}

	// Test 3: The point should be in the set (same x and y)
	if !ps.ContainsPoint(point3) {
		t.Errorf("Expected point %v to be in the ParetoSet", point3)
	}

	// Test 4: The point should NOT be in the set (different x and y)
	if ps.ContainsPoint(pointNotInSet) {
		t.Errorf("Expected point %v NOT to be in the ParetoSet", pointNotInSet)
	}

	// Test 5: Empty ParetoSet
	psEmpty := &ParetoSet{points: []Point{}}
	if psEmpty.ContainsPoint(point1) {
		t.Errorf("Expected point %v NOT to be in the empty ParetoSet", point1)
	}

	t.Log("Contains Method tested: ")
}

func TestInputCheck_NonNilPoint(t *testing.T) {
	ps := &ParetoSet{}
	p := &Point{x: 1.0, y: 2.0}

	result := ps.inputCheck(p)

	if result != p {
		t.Errorf("Expected inputCheck to return the same point, got %v", result)
	}

	t.Log("InputCheck test: ")
}

func TestInputCheck_NilPoint(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected inputCheck to panic on nil input, but it did not")
		}
	}()

	ps := &ParetoSet{}
	ps.inputCheck(nil) // Should panic

	t.Log("InputCheck test: ")

}

func TestMerge(t *testing.T) {
	ps1 := &ParetoSet{}
	ps1.Add(Point{x: 1.0, y: 5.0, Id: 1})
	ps1.Add(Point{x: 2.0, y: 3.0, Id: 2})

	ps2 := &ParetoSet{}
	ps2.Add(Point{x: 3.0, y: 6.0, Id: 3}) // dominates (2.0, 3.0)
	ps2.Add(Point{x: 4.0, y: 1.0, Id: 4}) // should be added

	Merge(ps1, ps2)

	expected := []*Point{
		{x: 1.0, y: 5.0},
		{x: 2.0, y: 3.0},
		{x: 4.0, y: 1.0},
	}

	for _, p := range expected {
		if !ps1.ContainsPoint(p) {
			t.Errorf("Expected point (%v, %v) not found after merge", p.x, p.y)
		}
	}
}

func TestParetoSet_Add(t *testing.T) {
	ps := &ParetoSet{}

	// Add initial Pareto point
	added := ps.Add(Point{x: 1.0, y: 5.0, Id: 1})
	if !added {
		t.Errorf("Expected point (1.0, 5.0) to be added")
	}
	if !ps.ContainsPoint(&Point{x: 1.0, y: 5.0}) {
		t.Errorf("Point (1.0, 5.0) should be in ParetoSet")
	}

	// Add dominated point (higher x, lower y)
	added = ps.Add(Point{x: 2.0, y: 3.0, Id: 2})
	if added {
		t.Errorf("Expected dominated point (2.0, 3.0) NOT to be added")
	}
	if ps.ContainsPoint(&Point{x: 2.0, y: 3.0}) {
		t.Errorf("Point (2.0, 3.0) should not be in ParetoSet")
	}

	// Add Pareto-optimal point
	added = ps.Add(Point{x: 3.0, y: 6.0, Id: 3})
	if !added {
		t.Errorf("Expected point (3.0, 6.0) to be added")
	}
	if !ps.ContainsPoint(&Point{x: 3.0, y: 6.0}) {
		t.Errorf("Point (3.0, 6.0) should be in ParetoSet")
	}

	// Add point that dominates the first one
	added = ps.Add(Point{x: 1.0, y: 7.0, Id: 4})
	if !added {
		t.Errorf("Expected point (1.0, 7.0) to be added and dominate previous (1.0, 5.0)")
	}
	if ps.ContainsPoint(&Point{x: 1.0, y: 5.0}) {
		t.Errorf("Old point (1.0, 5.0) should have been removed")
	}

	// Add duplicate point
	added = ps.Add(Point{x: 1.0, y: 7.0, Id: 5})
	if added {
		t.Errorf("Expected duplicate point (1.0, 7.0) NOT to be added again")
	}
}
