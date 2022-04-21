/*
    This file is using Coudert-Osmont Reduction Matrix (this is not a complete version, but a light one)
    His github : https://github.com/Nanored4498
*/

#include "common.h"

//#include "computeff.h"

#include "OpenNL_psm/OpenNL_psm.h"


using namespace std;


struct Coeff {
	static constexpr int    ONE = -2;
	static constexpr double EPS = 1e-8;

	Coeff(int p_index, double p_a) : index(p_index), a(p_a) {}
	Coeff(double const_value) : index(ONE), a(const_value) {}
	Coeff() : index(int(-1)), a(-1.0) {}

	inline Coeff operator-() const { return Coeff(index, -a); }
	inline bool is_null() const { return a == 0; }
	inline bool operator<(const Coeff &c) const { return index < c.index; }
	inline bool operator!=(const Coeff &c) const { return index != c.index || std::abs(a - c.a) > EPS; }
	inline friend Coeff operator*(double c, const Coeff& b) { return Coeff(b.index, c * b.a); }

	int index;
	double a;
};

struct LinExpr : public std::vector<Coeff> {
	LinExpr() : std::vector<Coeff>() { };
	LinExpr(int size, Coeff c) : std::vector<Coeff>(size, c) { };

	// Should not be needed
	void simplify(){
        std::sort(begin(), end());
        int s = 0;
        for (int i = 0; i < (int)size();) {
            operator[](s) = operator[](i++);
            while (i < size() && operator[](i).index == operator[](s).index)
                operator[](s).a += operator[](i++).a;
            if(std::abs(operator[](s).a) > Coeff::EPS) ++s;
        }
        resize(s);
    }

	LinExpr operator+(const LinExpr &e) const;
	LinExpr operator-(const LinExpr &e) const;
	inline LinExpr& operator+=(const LinExpr &e) { return (*this) = std::move(LinExpr(*this) + e); }
	inline LinExpr& operator-=(const LinExpr &e) { return (*this) = std::move(LinExpr(*this) - e); }

	LinExpr& operator+=(double x);
	LinExpr& operator*=(double x);
	inline LinExpr operator+(double x) const { return LinExpr(*this) += x; }
	inline friend LinExpr operator*(double x, const LinExpr &e) { return LinExpr(e) *= x; }

	bool operator==(const LinExpr &e) const;
	inline bool operator!=(const LinExpr &e) const { return !(*this == e); }

	friend std::ostream& operator<<(std::ostream& os, const LinExpr& obj);

	using std::vector<Coeff>::vector;
};

template<bool SUB>
LinExpr add(const LinExpr &e1, const LinExpr &e2) {
	LinExpr ans;
	int i = 0, j = 0;
	while(i < (int) e1.size() && j < (int) e2.size()) {
		if(e1[i].index < e2[j].index) ans.push_back(e1[i++]);
		else if(e1[i].index > e2[j].index) ans.push_back(SUB ? -e2[j++] : e2[j++]);
		else {
			const double a = SUB ? e1[i].a - e2[j].a : e1[i].a + e2[j].a;
			if(std::abs(a) > Coeff::EPS) ans.emplace_back(e1[i].index, a);
			++ i; ++ j;
		}
	}
	ans.reserve(ans.size() + e1.size() + e2.size() - (i+j));
	ans.insert(ans.end(), e1.begin()+i, e1.end());
	if(SUB) { while(j < (int)e2.size()) ans.push_back(-e2[j++]); }
	else ans.insert(ans.end(), e2.begin()+j, e2.end());
	return ans;
}
LinExpr LinExpr::operator+(const LinExpr &e) const { return add<false>(*this, e); }
LinExpr LinExpr::operator-(const LinExpr &e) const { return add<true>(*this, e); }
LinExpr& LinExpr::operator*=(double x) {
	if(x == 0.) clear();
	else {
		int s = 0;
		const double eps = Coeff::EPS/std::abs(x);
		for(int i = 0; i < (int) size(); ++i) if(std::abs((*this)[i].a) > eps)
			(*this)[s++] = x*(*this)[i];
		resize(s);
	}
	return *this;
}

struct SparseMatrix {
	std::vector<Coeff> mat;
	std::vector<int> offset;
	int m_ = -1;

	SparseMatrix() = default;
	SparseMatrix(int n, int m, int nCoeffs): mat(nCoeffs), offset(n+1), m_(m) { }

	inline int n() const { return int(offset.size())-1; }
	inline int m() const { return m_; }
	void clear();
	void resize(int n);

	struct rowRange {
		typedef std::vector<Coeff>::const_iterator iterator;
		iterator a, b;
		iterator begin() { return a; }
		iterator end() { return b; }
	};
	inline LinExpr row(int i) const { return LinExpr(mat.begin()+offset[i], mat.begin()+offset[i+1]); }
	inline rowRange rowIt(int i) const { return { mat.begin()+offset[i], mat.begin()+offset[i+1] }; }

	inline double eval(int i, const std::vector<double> &X) const { double y=0.; for(const Coeff &c : rowIt(i)) if(c.index == Coeff::ONE) y += c.a; else y += c.a * X[c.index]; return y; }

	void mult(const std::vector<double> &X, std::vector<double> &Y) const;
	inline std::vector<double> operator*(const std::vector<double> &X) const { std::vector<double> Y; mult(X, Y); return Y; }

	// Coeff::ONE are stored in the last row of the transpose matrix which has a shape (m+1, n)
	SparseMatrix transpose() const;
};

void SparseMatrix::mult(const std::vector<double> &X, std::vector<double> &Y) const {
	Y.assign(offset.size()-1, 0.);
	for(int i = 1, k = 0; i < (int) offset.size(); ++i)
		for(;k < offset[i]; ++k)
			if(mat[k].index == Coeff::ONE) Y[i-1] = mat[k].a;
			else Y[i-1] += mat[k].a * X[mat[k].index];
}

SparseMatrix SparseMatrix::transpose() const {
	SparseMatrix T(m()+1, n(), (int) mat.size());
	for(const Coeff &c : mat) if(c.index != Coeff::ONE) ++ T.offset[c.index+2];
	for(int i = 3; i < (int) T.offset.size(); ++i) T.offset[i] += T.offset[i-1];
	for(int i = 1, k = 0; i < (int) offset.size(); ++i)
		for(;k < offset[i]; ++k)
			T.mat[T.offset[mat[k].index == Coeff::ONE ? T.n() : mat[k].index+1]++] = Coeff(i-1, mat[k].a);
	return T;
}

struct ReductionBuilder {
	std::vector<LinExpr> lines;

	ReductionBuilder() = default;
	ReductionBuilder(int n): lines(n), refSize(n, 1) {
		for(int i = 0; i < n; ++i) lines[i].emplace_back(i, 1.);
	}
	ReductionBuilder(SparseMatrix &M, std::vector<int> &var_map, int newVars=0);

	inline int size() const { return lines.size(); }
	inline void clear() { lines.clear(); refSize.clear(); }
	void createNewVariables(int n);

	inline void sub(LinExpr &e) { sub(e, 0); }
	inline void simplify(int v) { if(!isVar(v)) sub(lines[v], refSize[v]); }
	void addEquality(LinExpr &e);
	inline void addEquality(LinExpr &&e) { addEquality(e); }
	inline bool isFixed(LinExpr &e) { sub(e); return e.empty() || (e.size() == 1 && e[0].index == Coeff::ONE); }

	SparseMatrix getMatrix();


private:
	std::vector<unsigned int> refSize;

	inline bool isVar(int v) { return v == Coeff::ONE || (lines[v].size() == 1 && lines[v][0].index == v); } 
	void sub(LinExpr &e, unsigned int ref);
	std::pair<int, int> compact();
};

ReductionBuilder::ReductionBuilder(SparseMatrix &M, std::vector<int> &var_map, int newVars): lines(M.n()+newVars), refSize(M.n()+newVars, 0) {
	var_map.clear();
	var_map.reserve(M.m());
	FOR(i, M.n()) if(M.offset[i+1] == M.offset[i]+1 && M.mat[M.offset[i]].index == (int) var_map.size())
		var_map.push_back(i);
	FOR(i, M.n()) {
		lines[i] = M.row(i);
		for(Coeff &c : lines[i]) if(c.index != Coeff::ONE) c.index = var_map[c.index];
	}
	for(int i = M.n(); i < size(); ++i) lines[i].emplace_back(i, 1.);
	for(const LinExpr &l : lines) for(const Coeff &c : l) if(c.index != Coeff::ONE) ++ refSize[c.index]; 
}

void ReductionBuilder::createNewVariables(int n) {
	int n0 = lines.size(), n2 = n0+n;
	lines.resize(n2);
	for(int i = n0; i < n2; ++i) lines[i].emplace_back(i, 1.);
	refSize.resize(lines.size(), 1u);
}

void ReductionBuilder::sub(LinExpr &e, unsigned int ref) {
	LinExpr new_e;
	for(const Coeff &c : e) {
		if(isVar(c.index)) new_e.push_back(c);
		else {
			sub(lines[c.index], refSize[c.index]);
			refSize[c.index] -= ref;
			for(const Coeff &c2 : lines[c.index])
				new_e.emplace_back(c2.index, c.a * c2.a);
		}
	}
	std::sort(new_e.begin(), new_e.end());
	int s = 0;
	for(int i = 0; i < (int) new_e.size();) {
		new_e[s] = new_e[i++];
		while(i < new_e.size() && new_e[i].index == new_e[s].index) {
			new_e[s].a += new_e[i++].a;
			if(new_e[s].index != Coeff::ONE) refSize[new_e[s].index] -= ref;
		}
		if(std::abs(new_e[s].a) > Coeff::EPS) ++ s;
		else if(new_e[s].index != Coeff::ONE) refSize[new_e[s].index] -= ref;
	}
	new_e.resize(s);
	e = std::move(new_e);
}

void ReductionBuilder::addEquality(LinExpr &e) {
	sub(e);
	if(e.empty()) return;
	if(e.size() == 1 && e[0].index == Coeff::ONE) return ;//throw ImpossibleEquality(e[0].a);
	int v = e[0].index == Coeff::ONE ? 1 : 0;
	for(int w = v+1; w < (int) e.size(); ++w)
		if(refSize[e[w].index] < refSize[e[v].index])
			v = w;
	// int v = e[0].index != Coeff::ONE ? rand() % e.ts.size() : 1 + (rand() % (e.ts.size()-1));
	if(v+1 != (int) e.size()) std::swap(e[v], e.back());
	v = e.back().index;
	double mul = - 1. / e.back().a;
	e.pop_back();
	sort(e.begin(), e.end());
	for(Coeff &c : e) {
		c.a *= mul;
		if(c.index != Coeff::ONE) refSize[c.index] += refSize[v];
	}
	lines[v] = std::move(e);
}

std::pair<int, int> ReductionBuilder::compact() {
	int m = 0, nCoeffs = 0;
	FOR(v, size()) simplify(v);
	FOR(v, size())
		if(isVar(v)) lines[v][0].index = m++;
		else for(Coeff &c : lines[v]) if(c.index != Coeff::ONE && c.index < v) c.index = lines[c.index][0].index;
	FOR(v, size()) {
		nCoeffs += lines[v].size();
		for(Coeff &c : lines[v]) if(c.index > v) c.index = lines[c.index][0].index;
	}
	return { m, nCoeffs };
}

SparseMatrix ReductionBuilder::getMatrix() {
	auto [m, nCoeffs] = compact();
	SparseMatrix M(lines.size(), m, nCoeffs);
	M.offset[0] = 0;
	for(int i = 1; i <= lines.size(); ++i) {
		M.offset[i] = M.offset[i-1];
		for(const Coeff &c : lines[i-1]) M.mat[M.offset[i]++] = c;
	}
	clear();
	return M;
}

struct LSSolver {
	LSSolver() = default;
	LSSolver(int p_nb_variables, bool verbose = false, double eps = 1e-10, int nb_max_iter = 5000);
	void init(int p_nb_variables, bool verbose=false, double eps=1e-10, int nb_max_iter=5000);
	void add_to_energy(const LinExpr &line);
	void fix(int var, double value);
	void solve();
	std::vector<double> X;
private:
	void* context;
	bool fix_var_step_done;
};

LSSolver::LSSolver(int p_nb_variables, bool verbose, double eps, int nb_max_iter) { init(p_nb_variables, verbose, eps, nb_max_iter); }
void LSSolver::init(int p_nb_variables,bool verbose,double eps,int nb_max_iter) {
	fix_var_step_done = false;
	X.resize(p_nb_variables, 0.);
	context = nlNewContext();
	if (verbose) nlEnable(NL_VERBOSE);
	nlEnable(NL_VARIABLES_BUFFER);
	nlSolverParameterd(NL_THRESHOLD, eps);
	nlSolverParameteri(NL_MAX_ITERATIONS, nb_max_iter);
	nlSolverParameteri(NL_LEAST_SQUARES, NL_TRUE);
	nlSolverParameteri(NL_NB_VARIABLES, NLint(X.size()));
	nlBegin(NL_SYSTEM);
	nlBindBuffer(NL_VARIABLES_BUFFER, 0, (void*) X.data(), (NLuint) sizeof(X[0]));
}

void LSSolver::fix(int var, double value) {
	X[var] = value;
	nlMakeCurrent(context);
	nlLockVariable(NLint(var));
}

void LSSolver::add_to_energy(const LinExpr &line) {
	nlMakeCurrent(context);
	if(!fix_var_step_done) {
		nlBegin(NL_MATRIX);
		fix_var_step_done = true;
	}
	nlBegin(NL_ROW);
	double rhs = 0.;
	for(const Coeff &c : line)
		if(c.index == Coeff::ONE) rhs -= c.a;
		else nlCoefficient(c.index, c.a);
	nlRightHandSide(rhs);
	nlEnd(NL_ROW);
}

void LSSolver::solve() {
	nlMakeCurrent(context);
	nlEnd(NL_MATRIX);
	nlEnd(NL_SYSTEM);
	nlSolve();
	nlDeleteContext(context);
	context = nullptr;
}


struct vec2Expr {
	LinExpr x, y;

	vec2Expr() = default;
	vec2Expr(const LinExpr &x, const LinExpr &y): x(x), y(y) {}

	inline LinExpr& operator[](int d) { return reinterpret_cast<LinExpr*>(this)[d]; }
	inline const LinExpr& operator[](int d) const { return reinterpret_cast<const LinExpr*>(this)[d]; }

	inline vec2Expr operator+(const vec2Expr &v) const { return { x + v.x, y + v.y }; }
	inline vec2Expr operator-(const vec2Expr &v) const { return { x - v.x, y - v.y }; }
	inline vec2Expr& operator+=(const vec2Expr &v) { x += v.x; y += v.y; return *this; }
	inline vec2Expr& operator-=(const vec2Expr &v) { x -= v.x; y -= v.y; return *this; }

	vec2Expr& rotate(int k) {
        if(k&1) std::swap(x, y);
        if(k&2) y *= -1.;
        if((k+1)&2) x *= -1.;
        return *this;
    }
};

vector<int> compute_jumps0(const TriMesh& m, const TriConnectivity& tc, std::vector<vec2> &alpha) {

	std::vector<double> angleInFacet(m.nh(), 0.);
	FOR(f, m.nf()) {
		vec3 ref = normalized(tc.geom(3 * f));
		angleInFacet[3 * f + 1] = std::acos(dot(ref, normalized(tc.geom(3 * f + 1))));
		angleInFacet[3 * f + 2] = -std::acos(dot(ref, normalized(tc.geom(3 * f + 2))));
	}

	//param.zeroPJ.ptr->data.assign(param.m.ncorners(), false);
    vector<int> valence(m.points.size());
    vector<int> jumpPJ(m.nh(), 0);
	std::vector<bool> seen(m.nf(), false);
	std::vector<int> Qring;
	std::vector<int> nz(m.points.size(), 0);
	for(int v : m.h2v) ++ nz[v];
	const auto setZero = [&](int h)->void {
		jumpPJ[h] |= 1;
		int v = tc.from(h);
		if(--nz[v] == 1) Qring.push_back(v);
	};
	prln("Avant qtree :");
	for(int f_seed = 0; f_seed < m.nf(); ++f_seed) if(!seen[f_seed]) {
		std::queue<int> Qtree;
		seen[f_seed] = true;
		Qtree.push(f_seed);
		while(!Qtree.empty()) {
			int f = Qtree.front();
			Qtree.pop();
			for(int fc = 0; fc < 3; ++fc) {
				int h = 3 * f + fc;
				int h2 = tc.opp(h);
				if(h2 == -1) {
					setZero(h);
					continue;
				}
				const int f2 = h2/3;
				const double theta = angleInFacet[h] - angleInFacet[h2] + M_PI;
				double best = 1e9;
				vec2 af = alpha[f] - vec2({theta, theta});
				FOR(j, 4) {
					const vec2 diff = (af - alpha[f2]) * (1. / (2. * M_PI));
					const double p = std::round(.5 * (diff.x + diff.y));
					double d = std::abs(diff.x - p) + std::abs(diff.y - p);
					d = std::min(d, std::abs(diff.x - p-1) + std::abs(diff.y - p-1));
					d = std::min(d, std::abs(diff.x - p+1) + std::abs(diff.y - p+1));
					if(d < best) {
						best = d;
						jumpPJ[h] = jumpPJ[h]&1;
						jumpPJ[h] |= j << 1;
					}
					std::swap(af.x, af.y);
					af.y += M_PI;
				}
				if(!seen[f2]) {
					if((jumpPJ[h] >> 1)&1) std::swap(alpha[f2][0], alpha[f2][1]);
					if(((jumpPJ[h] >> 1)+1)&2) alpha[f2][0] -= M_PI;
					if((jumpPJ[h] >> 1)&2) alpha[f2][1] -= M_PI;
					if((jumpPJ[h] >> 1)==3) alpha[f2][0] -= 2.*M_PI;
					setZero(h);
					setZero(h2);
					seen[f2] = true;
					Qtree.push(f2);
				}
			}
		}
	}
	prln("Valence :");
    // Compute valences
	FOR(v, m.points.size()) if(tc.v2h[v] != -1){
		int h = tc.v2h[v], bh=-1, obh=-1;
		double ind = 2.*M_PI;
		do {
			const int ph = tc.prev(h);
			const double corner_angle = M_PI + angleInFacet[ph] - angleInFacet[h];
			ind += 2*M_PI * round(corner_angle / (2.*M_PI)) - corner_angle;
			const int oh = tc.opp(h);
			if(oh == -1) bh = h;
			else {
				const int f = tc.facet(h), of = tc.facet(oh);
				const double theta = angleInFacet[h] - angleInFacet[oh] + M_PI;
				const double rot = alpha[f][0] - alpha[of][(jumpPJ[h] >> 1) &1] - theta;
				const double pj = 2. * round(rot / M_PI) + ((jumpPJ[h] >> 1)&1);
				ind -= theta + pj*M_PI_2;
			}
			if(tc.opp(ph) == -1) obh = ph;
			h = tc.h2h[h];
		} while(h != -1 && h != tc.v2h[v]);
		if(bh != -1) {
			const int f = tc.facet(bh), of = tc.facet(obh);
			const double theta = angleInFacet[bh] - angleInFacet[obh] + M_PI;
			const double rot = alpha[f][0] - alpha[of][0] - theta;
			ind -= theta + round(rot / M_PI_2)*M_PI_2;
		}
		ind /= 2.*M_PI;
		valence[v] = 4 - round(4.*ind);
	}

	prln("Simplify cut graph");
	// Remove some edges from cut graph
	while(!Qring.empty()) {
		int v = Qring.back();
		Qring.pop_back();
		if(nz[v] == 0) continue;
		if(valence[v] != 4 || tc.is_boundary_vert(v)) continue;
		int h = tc.v2h[v];
		while(jumpPJ[h] & 1){
			h = tc.h2h[h];
		}
		setZero(h);
		int h2 = tc.opp(h);
		setZero(h2);
	}
	prln("Bye !");
	return jumpPJ;
}

vector<vec2> angles_to_ffuv(const TriMesh& m, const TriConnectivity& tc, const vector<vec2>& alpha, double desired_size = 1){
    vector<vec2> ff_uv(m.nh());
    FOR(f, m.nf()){
        ff_uv[3 * f] = vec2{0, 0};
		const double ca = std::cos(alpha[f][0]), sa = std::sin(alpha[f][0]);
		const double cb = std::cos(alpha[f][1]), sb = std::sin(alpha[f][1]);
		const double l0 = norm(tc.geom(3 * f));
		ff_uv[3 * f + 1] = vec2{ ca * l0, cb * l0 } / desired_size;
		const double l1 = - norm(tc.geom(3 * f + 2));
        double angle_ref  = -std::acos(dot(normalized(tc.geom(3 * f)), normalized(tc.geom(3 * f + 2))));
		const double c1 = l1 * cos(angle_ref), s1 = l1 * sin(angle_ref);
        ff_uv[3 * f + 2] = vec2{ ca * c1 + sa * s1, cb * c1 + sb * s1 } / desired_size;
    }
    return ff_uv;
}

ReductionBuilder initRB(const TriMesh& m, const TriConnectivity& tc, const vector<vec2>& alpha, const vector<int>& jumpPJ) {
    auto ffuv = [&alpha, &tc] (int h) {
        if(h % 3 == 0) return vec2{0, 0};
		const double ca = std::cos(alpha[h/3][0]), sa = std::sin(alpha[h/3][0]);
		const double cb = std::cos(alpha[h/3][1]), sb = std::sin(alpha[h/3][1]);
        if(h % 3 == 1){
		    const double l0 = norm(tc.geom(h - 1));
		    return vec2{ ca * l0, cb * l0 };
        }
		const double l1 = - norm(tc.geom(h));
        double angle_ref  = -std::acos(dot(normalized(tc.geom(h - 2)), normalized(tc.geom(h))));
		const double c1 = l1 * cos(angle_ref), s1 = l1 * sin(angle_ref);
        return vec2{ ca * c1 + sa * s1, cb * c1 + sb * s1 };
	};

	prln("Initialization :");
	// Create system
	ReductionBuilder builder(m.nh() << 1);
    vector<bool> featureEdge = compute_is_feature(m, tc);

	
	prln("First step :");
	// Add equalities
	for(int h = 0; h < m.nh(); ++h) {
		int opp = tc.opp(h);
		if(featureEdge[h] && opp < h) {
			const vec2 duv = ffuv(tc.next(h)) - ffuv(h);
			const int d = std::abs(duv.x) < std::abs(duv.y) ? 0 : 1;
			builder.addEquality(builder.lines[(h << 1) | d] - builder.lines[(tc.next(h) << 1) | d]);
		}
		if(opp == -1 || !(jumpPJ[h]&1)) continue;
		opp = tc.next(opp);
		FOR(d, 2) builder.addEquality(builder.lines[2*h+d] - builder.lines[2*opp+d]);
	}

	prln("Second step :");
	// Second step
	FOR(v, builder.size()) builder.simplify(v);
	for(int h = 0; h < m.nh(); ++h) {
		builder.simplify(2*h);
		builder.simplify(2*h+1);
		if(jumpPJ[h]&1) continue;
		const int opp = tc.opp(h);
		if(opp < h) continue;
		const int nh = tc.next(h);
		const int nopp = tc.next(opp);
		vec2Expr oppVec(builder.lines[2*nopp] - builder.lines[2*opp], builder.lines[2*nopp+1] - builder.lines[2*opp+1]);
		oppVec.rotate(jumpPJ[h] >> 1);
		builder.addEquality(builder.lines[2*nh] - builder.lines[2*h] + oppVec.x);
		builder.addEquality(builder.lines[2*nh+1] - builder.lines[2*h+1] + oppVec.y);
	}

    return builder;
    /*
	auto M = builder.getMatrix();
	int ml = 0;
	FOR(i, M.n()) ml = std::max(ml, M.offset[i+1] - M.offset[i]);
	std::cerr << M.m() << " / " << M.n() << "  nverts: " << m.nverts() << "  max len: " << ml << "  nCoeffs: " << M.mat.size() << std::endl;
    */
}

vector<vec2> computeUV(const TriMesh& m, const TriConnectivity& tc, const SparseMatrix& M, const vector<vec2>& ffuv) {
    auto getGradMat = [&m, &tc] (int f) {
        double Gr[2][2];
        const vec3 e0 = tc.geom(3*f);
        const vec3 e2 = tc.geom(3*f+2);
        const double x1 = norm(e0);
        const double x2 = - dot(e0, e2) / x1;
        const double y2 = std::sqrt(norm2(e2) - x2*x2);
        double area = x1 * y2;
        Gr[0][0] = y2/area; Gr[0][1] = 0.;
        Gr[1][0] = -x2/area; Gr[1][1] = x1/area;
        area *= .5;
        return make_pair(Gr, sqrt(area));
    };
	LSSolver solver(M.m());
	//LSSolver solver = initRB(m, tc, alpha, compute_jumps0(m, tc, alpha));
	/*
	for(int f = 0; f < m.nf(); ++f) {
		auto [Gr, sqrt_area] = getGradMat(f);
		for(int i : {0, 1}) for(int j : {0, 1}) Gr[i][j] *= sqrt_area;
		for(int d : {0, 1}) {
			const LinExpr uab = M.row(6*f+2+d) - M.row(6*f+d);
			const LinExpr uac = M.row(6*f+4+d) - M.row(6*f+d);
			for(int i : {0, 1}) {
				LinExpr line = Gr[i][0] * uab + Gr[i][1] * uac;
				line.emplace_back(Coeff::ONE, - (Gr[i][0] * (ffuv[3*f+1] - ffuv[3*f])[d] + Gr[i][1] * (ffuv[3*f+1] - ffuv[3*f])[d]));
				solver.add_to_energy(line);
			}
		}
	}*/
	FOR(h, m.nh()) FOR(d, 2){
		LinExpr line = M.row(tc.next(h) << 1 | d) - M.row(h << 1 | d);
		line.emplace_back(Coeff::ONE, - (ffuv[tc.next(h)] - ffuv[h])[d]);
		solver.add_to_energy(line);
	}
	solver.solve();
    vector<vec2> U(m.nh());
	FOR(d, 2) FOR(h, m.nh()) U[h][d] = M.eval(2*h+d, solver.X);
	//computeOrientation();

	return U;
}

vector<double> compute_FF_angles(const TriMesh& m, const TriConnectivity& tc, const vector<bool>& is_feature);
vector<double> compute_param(const vector<int>& ph2v, const vector<double>& ppoints){ //, const vector<double>& ff_angles
	clock_t param_start = clock();
    TriMesh m; m.h2v = ph2v;
    m.points.resize(ppoints.size()/3);
    FOR(i, ppoints.size()) m.points[i/3][i%3] = ppoints[i]; 
    TriConnectivity tc(m);
    FOR(h, m.nh()) if(tc.opp(h) == -1) {prln("There is boundary verts"); break;}
    vector<bool> is_feature(m.nh(), false);
	
	const vector<double> ff_angles = compute_FF_angles(m, tc, compute_is_feature(m, tc));
    std::vector<vec2> alpha(m.nf());
	FOR(f, m.nf()) alpha[f] = { ff_angles[f], ff_angles[f] + M_PI_2 };
	double edge_size = 0; FOR(h, m.nh()) edge_size += norm(tc.geom(h));  edge_size /= m.nh();
    //FOR(f, m.nf()) alpha[f] = { ff_angles[f << 1], ff_angles[(f << 1) | 1] };
	prln("Builder computation :");
    ReductionBuilder builder = initRB(m, tc, alpha, compute_jumps0(m, tc, alpha));
	auto M = builder.getMatrix();
	int ml = 0;
	FOR(i, M.n()) ml = std::max(ml, M.offset[i+1] - M.offset[i]);
	std::cerr << M.m() << " / " << M.n() << "  nverts: " << m.nh() << "  max len: " << ml << "  nCoeffs: " << M.mat.size() << std::endl;
	prln("U coordinates computation :");
    vector<vec2> ffuv = angles_to_ffuv(m, tc, alpha, edge_size);//computeUV(m, tc, builder.getMatrix(), angles_to_ffuv(m, tc, alpha, edge_size));
	vector<vec2> U = computeUV(m, tc, M, ffuv);
	prln("Done !");
	//vector<vec2> U = angles_to_ffuv(m, tc, alpha, edge_size);


    vector<double> texcoords(m.nh() << 1);
    FOR(h, m.nh()) FOR(i, 2) texcoords[h << 1 | i] = U[h][i];
	
    clock_t param_end = clock(); double time_taken = double(param_end - param_start) / double(CLOCKS_PER_SEC);
	cout << "Computation of param took time : "  << time_taken*1000 << "\n";

    return texcoords;
}