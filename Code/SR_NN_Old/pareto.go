package main

import (
	"fmt"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"log"
	"math"
	"math/rand"
	"sort"
)

type Point struct {
	x    float64
	y    float64
	Data interface{}
	Id   int
}

type ParetoSet struct {
	points []Point
}

func NewParetoSet() *ParetoSet {
	return &ParetoSet{
		points: []Point{},
	}
}

// Get returns the value at a given index: 0 -> X, 1 -> Y, 2 -> Data, 3 -> Id
func (p *Point) Get(index int) interface{} {
	switch index {
	case 0:
		return p.x
	case 1:
		return p.y
	case 2:
		return p.Data
	case 3:
		return p.Id
	default:
		panic(fmt.Sprintf("Index %d out of range!", index))
	}
}

func (p *Point) Set(index int, value interface{}) {
	switch index {
	case 0:
		p.x = value.(float64)
	case 1:
		p.y = value.(float64)
	case 2:
		p.Data = value
	case 3:
		panic("Cannot set ID!")
	default:
		panic(fmt.Sprintf("Index %d out of range!", index))
	}
}

func (ps *ParetoSet) Add(p Point) bool {
	isPareto := false

	// Binary search to find insertion index based on X
	right := ps.bisectLeft(p)

	// Remove dominated points on the right
	for right < len(ps.points) &&
		ps.points[right].y >= p.y &&
		!(ps.points[right].x == p.x && ps.points[right].y == p.y) {

		ps.points = append(ps.points[:right], ps.points[right+1:]...)
		isPareto = true
	}

	// Check if any point on the left dominates this point
	left := ps.bisectRight(p) - 1

	if left == -1 || ps.points[left].y > p.y {
		isPareto = true
	}

	if len(ps.points) == 0 {
		isPareto = true
	}

	if isPareto {
		// Maintain order by inserting at the correct position
		ps.points = append(ps.points, Point{})       // Make space
		copy(ps.points[right+1:], ps.points[right:]) // Shift elements
		ps.points[right] = p                         // Insert point
	}

	return isPareto
}

func (ps *ParetoSet) bisectLeft(p Point) int {
	low, high := 0, len(ps.points)
	for low < high {
		mid := (low + high) / 2
		if ps.points[mid].x < p.x {
			low = mid + 1
		} else {
			high = mid
		}
	}
	return low
}

func (ps *ParetoSet) bisectRight(p Point) int {
	low, high := 0, len(ps.points)
	for low < high {
		mid := (low + high) / 2
		if ps.points[mid].x <= p.x {
			low = mid + 1
		} else {
			high = mid
		}
	}
	return low
}

// gives you minimum distance from two points
func (ps *ParetoSet) Distance(p Point) float64 {
	dominating := ps.getDominantPoints(p)

	if len(dominating) == 0 {
		return 0.0
	}

	candidates := make([][]float64, len(dominating)+1)

	for i := 0; i < len(dominating)-1; i++ {
		candidates[i] = []float64{
			math.Max(dominating[i].x, dominating[i+1].x),
			math.Max(dominating[i].y, dominating[i+1].y),
		}
	}

	candidates[len(candidates)-2] = []float64{p.x, minY(dominating)}
	candidates[len(candidates)-1] = []float64{minX(dominating), p.y}

	minDist := math.Inf(1)
	for _, c := range candidates {
		dx := p.x - c[0]
		dy := p.y - c[1]
		dist := math.Sqrt(dx*dx + dy*dy)
		if dist < minDist {
			minDist = dist
		}
	}
	return minDist
}

func (ps *ParetoSet) getDominantPoints(p Point) []Point {
	var dom []Point
	for i := len(ps.points) - 1; i >= 0; i-- {
		if ps.points[i].x < p.x && ps.points[i].y < p.y {
			dom = append(dom, ps.points[i])
		}
	}
	return dom
}

// Contains checks if a point with same x and y exists
func (ps *ParetoSet) Contains(p Point) bool {
	for _, point := range ps.points {
		if point.x == p.x && point.y == p.y {
			return true
		}
	}
	return false
}

// ToArray returns the Pareto set as a slice of [x, y, data]
func (ps *ParetoSet) ToArray() [][]interface{} {
	out := make([][]interface{}, len(ps.points))
	for i, p := range ps.points {
		out[i] = []interface{}{p.x, p.y, p.Data}
	}
	return out
}

// Merge takes another ParetoSet and adds all of its points to the current set.
func Merge(ps1, ps2 *ParetoSet) {
	for _, p := range ps2.points {
		ps1.Add(p)
	}
}

// gets all id and returns it
func (ps *ParetoSet) GetIdList() []int {
	ids := make([]int, len(ps.points))
	for i, p := range ps.points {
		ids[i] = p.Id
	}
	return ids
}

// ToXYArray returns a 2D slice containing [X, Y] for all points in the ParetoSet.
func (ps *ParetoSet) ToXYArray() [][]float64 {
	out := make([][]float64, len(ps.points))
	for i, p := range ps.points {
		out[i] = []float64{p.x, p.y}
	}
	return out
}

// GetParetoPoints returns a 2D slice where each inner slice contains [X, Y, Data] for a point.
func (ps *ParetoSet) GetParetoPoints() [][]interface{} {
	out := make([][]interface{}, len(ps.points))
	for i, p := range ps.points {
		out[i] = []interface{}{p.x, p.y, p.Data}
	}
	return out
}

// FromList takes a slice of *Point and adds them to the ParetoSet.
func (ps *ParetoSet) FromList(points []*Point) {
	for _, p := range points {
		ps.Add(*p)
	}
}

// Plot generates a scatter plot of the Pareto frontier and saves it as a PNG.
func (ps *ParetoSet) Plot(filename string) {
	pts := make(plotter.XYs, len(ps.points))
	for i, p := range ps.points {
		pts[i].X = p.x
		pts[i].Y = p.y
	}

	p := plot.New()

	p.Title.Text = "Pareto Frontier"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	scatter, err := plotter.NewScatter(pts)
	if err != nil {
		log.Fatalf("failed to create scatter: %v", err)
	}

	p.Add(scatter)

	if err := p.Save(8*vg.Inch, 6*vg.Inch, filename); err != nil {
		log.Fatalf("failed to save plot: %v", err)
	}
}

// Contains checks whether the given Point is already in the ParetoSet.
func (ps *ParetoSet) ContainsPoint(p *Point) bool {
	p = ps.inputCheck(p)

	// Find where p would be inserted
	left := ps.bisectLeft(*p)

	// Scan forward to see if a point with same X and Y exists
	for left < len(ps.points) && ps.points[left].x == p.x {
		if ps.points[left].y == p.y {
			return true
		}
		left++
	}

	return false
}

// DominantArray returns a slice of [X, Y] values for all points in the set that dominate point p.
func (ps *ParetoSet) DominantArray(p *Point) [][]float64 {
	p = ps.inputCheck(p)

	idx := ps.bisectLeft(*p) - 1
	var domList [][]float64

	for idx >= 0 && ps.points[idx].y < p.y {
		domList = append(domList, []float64{ps.points[idx].x, ps.points[idx].y})
		idx--
	}

	return domList
}

// inputCheck ensures the point is non-nil.
func (ps *ParetoSet) inputCheck(p *Point) *Point {
	if p == nil {
		panic("inputCheck: received nil Point")
	}
	return p
}

// bisectLeft returns the index where point p should be inserted to maintain sorted order by X.
func (ps *ParetoSet) bisectLeft1(p *Point) int {
	x := p.x
	return sort.Search(len(ps.points), func(i int) bool {
		return ps.points[i].x >= x
	})
}

func minY(points []Point) float64 {
	min := math.Inf(1)
	for _, p := range points {
		if p.y < min {
			min = p.y
		}
	}
	return min
}

func minX(points []Point) float64 {
	min := math.Inf(1)
	for _, p := range points {
		if p.x < min {
			min = p.x
		}
	}
	return min
}

func main() {
	ps := NewParetoSet()
	for i := 0; i < 40; i++ {
		x := rand.Float64()
		y := rand.Float64()
		p := Point{x: x, y: y}
		ps.Add(p)
	}

	for _, pt := range ps.points {
		fmt.Printf("x: %.4f, y: %.4f\n", pt.x, pt.y)
	}
}
