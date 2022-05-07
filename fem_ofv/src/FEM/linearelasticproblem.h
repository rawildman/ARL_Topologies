/*
 * ARL_Topologies - An extensible topology optimization program
 * 
 * Written in 2022 by Raymond A. Wildman 
 * 
 * To the extent possible under law, the author(s) have dedicated all copyright 
 * and related and neighboring rights to this software to the public domain 
 * worldwide. This software is distributed without any warranty.
 * 
 * You should have received a copy of the CC0 Public Domain Dedication along 
 * with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>. 
 * 
 */

#ifndef LINEAR_ELASTIC_PROBLEM_H
#define LINEAR_ELASTIC_PROBLEM_H

#include "femproblem.h"

//! A class that sets up and solves a static, linear elastic finite element problem
/*! This class takes a mesh and a base material, and sets up its own finite element mesh, 
 *  which can compute element matrices and other values needed to solve an FEM problem.  
 *  The base material baseMat is modified by the optVal parameters in TOMesh.
*/
class LinearElasticProblem : public FEMProblem
{
public:
	LinearElasticProblem(const Topologies::TOMesh& inMesh, 
		const Topologies::GenericMaterial& baseMat, 
		const std::vector<MaterialFunction>& optimizationToMaterialFuns);

	void changeBoundaryConditionsTo(const std::vector<ExoBC>& bcVec) override;
	std::pair<double, bool> computeCompliance() override;
	std::vector<double> gradCompliance(const Topologies::TOMesh& inMesh) const override;
private:
	typedef Eigen::MatrixXd EigenDenseMat;
	typedef Eigen::VectorXd EigenVector;
	typedef Eigen::SparseMatrix<double> EigenSparseMat;
	typedef Eigen::Triplet<double> EigenT;

	void solveProblem();
	void setMatrix();
	void setVector();
	void assembleMatrix(std::vector<EigenT>& rseMat, const std::size_t kelem, const EigenDenseMat& elemMat, std::size_t numUnk) const;
	void assembleMatrix2D(std::vector<EigenT>& rseMat, const std::size_t kelem, const EigenDenseMat& elemMat, std::size_t numUnk) const;
	void assembleMatrix3D(std::vector<EigenT>& rseMat, const std::size_t kelem, const EigenDenseMat& elemMat, std::size_t numUnk) const;
	double elementCompliance(const std::size_t kelem, const EigenDenseMat& elemMat, std::size_t numUnk) const;
	double elementCompliance2D(const std::size_t kelem, const EigenDenseMat& elemMat, std::size_t numUnk) const;
	double elementCompliance3D(const std::size_t kelem, const EigenDenseMat& elemMat, std::size_t numUnk) const;
};

#endif