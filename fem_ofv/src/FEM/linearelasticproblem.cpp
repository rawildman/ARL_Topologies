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

#include "linearelasticproblem.h"

LinearElasticProblem::LinearElasticProblem(const Topologies::TOMesh &inMesh,
	const Topologies::GenericMaterial &baseMat,
	const std::vector<MaterialFunction>& optimizationToMaterialFuns) 
	: FEMProblem(inMesh, baseMat, optimizationToMaterialFuns)
{
}

void LinearElasticProblem::changeBoundaryConditionsTo(const std::vector<ExoBC>& bcVec)
{
	std::size_t nnodes = probMesh->getNumUnknowns();
	fixedDOFs = std::vector<bool>(dim*nnodes, false);
	loadVec.clear();
	for(auto const& curBC : bcVec)
	{
		for(auto kn : curBC.nodeIDVec)
		{
			// Set up supports
			if(curBC.isSupport)
			{
				fixedDOFs[kn] = curBC.xsup;
				fixedDOFs[kn + nnodes] = curBC.ysup;
				if(dim == 3)
					fixedDOFs[kn + 2*nnodes] = curBC.zsup;
			}
			else // Set up loads
			{
				Point3D cartVec = CoordinateSystem::convertVector(curBC.loadVec, getMeshPoint(kn), curBC.ct);
				loadVec[kn] = cartVec.x;
				loadVec[kn + nnodes] = cartVec.y;
				if(dim == 3)
					loadVec[kn + 2*nnodes] = cartVec.z;
			}
		}
	}
	// Save global row numbers for compressed system (without fixed dofs)
	bfRemapVec.resize(dim*nnodes);
	std::size_t curBF = 0;
	for(std::size_t k = 0; k < fixedDOFs.size(); ++k)
	{
		if(!fixedDOFs[k])
			bfRemapVec[k] = curBF++;
	}
	numFreeDOFs = curBF;
	solveProblem();
}

void LinearElasticProblem::solveProblem()
{
	// Set up vectors and matrices
	setMatrix();
	setVector();
	// Matrix stuff:
	// Save RHS to use for computation of compliance
	pForce = std::make_unique<EigenVector>(*pVVec);
	invalid = false;
	// Solve
	Eigen::ConjugateGradient<EigenSparseMat, Eigen::Lower | Eigen::Upper> solver;
	solver.compute(*pFEMMatrix);
	solver.setTolerance(itTol);
	unsigned niters = 10000;
	solver.setMaxIterations(niters);
	*pVVec = solver.solve(*pForce);
	if(solver.iterations() >= niters)
  {
    std::cout << "Warning: Solver didn't converge" << std::endl;
		invalid = true;
  }
}

void LinearElasticProblem::setMatrix()
{
	std::size_t numNodes = probMesh->getNumUnknowns();
	std::vector<EigenT> sparseVec;
	sparseVec.reserve(10*numFreeDOFs); //Size of this depends on matrix bandwidth, which is unknown as it depends on mesh connectivity
  std::size_t nelems = probMesh->getNumElements();
  for(std::size_t ielem = 0; ielem < nelems; ++ielem)
  {
    Eigen::MatrixXd elemMat = probMesh->getElementMatrix(ielem);
    assembleMatrix(sparseVec, ielem, elemMat, numNodes);
  }
	pFEMMatrix = std::make_unique<EigenSparseMat>(numFreeDOFs,numFreeDOFs);
	pFEMMatrix->setFromTriplets(sparseVec.begin(), sparseVec.end());
}

void LinearElasticProblem::assembleMatrix(std::vector<EigenT>& rseMat, const std::size_t kelem, const Eigen::MatrixXd& elemMat, std::size_t numUnk) const
{
	if(dim == 2)
		assembleMatrix2D(rseMat, kelem, elemMat, numUnk);
	else if(dim == 3)
		assembleMatrix3D(rseMat, kelem, elemMat, numUnk);
}

void LinearElasticProblem::assembleMatrix2D(std::vector<EigenT>& rseMat, const std::size_t kelem, const Eigen::MatrixXd& elemMat, std::size_t numUnk) const
{
	const unsigned numElemNodes = probMesh->getNumElementNodes(kelem);
	for(unsigned iTst = 0; iTst < numElemNodes; iTst++)
	{
		std::size_t gTst = probMesh->getGlobalBF(kelem, iTst);
		for(unsigned iBas = 0; iBas < numElemNodes; iBas++)
		{
			std::size_t gBas = probMesh->getGlobalBF(kelem, iBas);
			for(unsigned tstdof = 0; tstdof < dim; ++tstdof)
			{
				for(unsigned basdof = 0; basdof < dim; ++basdof)
				{
					bool isFree = !(fixedDOFs[gTst + tstdof*numUnk] || fixedDOFs[gBas + basdof*numUnk]);
					if(fabs(elemMat(iTst + tstdof*numElemNodes, iBas + basdof*numElemNodes)) > 1.e-16 && isFree)
					{
						rseMat.push_back(EigenT(bfRemapVec[gTst + tstdof*numUnk],
																		bfRemapVec[gBas + basdof*numUnk],
																		elemMat(iTst + tstdof*numElemNodes, iBas + basdof*numElemNodes)));
					}
				}
			}
		}
	}
}

void LinearElasticProblem::assembleMatrix3D(std::vector<EigenT>& rseMat, const std::size_t kelem, const Eigen::MatrixXd& elemMat, std::size_t numUnk) const
{
	const unsigned numElemNodes = probMesh->getNumElementNodes(kelem);
	for(unsigned iTst = 0; iTst < numElemNodes; iTst++)
	{
		std::size_t gTst = probMesh->getGlobalBF(kelem, iTst);
		for(unsigned iBas = 0; iBas < numElemNodes; iBas++)
		{
			std::size_t gBas = probMesh->getGlobalBF(kelem, iBas);
			for(unsigned tstOff = 0; tstOff < dim; ++tstOff)
			{
				for(unsigned basOff = 0; basOff < dim; ++basOff)
				{
					bool isFree = !(fixedDOFs[gTst + tstOff*numUnk] || fixedDOFs[gBas + basOff*numUnk]);
					if(fabs(elemMat(dim*iTst + tstOff, dim*iBas + basOff)) > 1.e-16 && isFree)
					{
						rseMat.push_back(EigenT(bfRemapVec[gTst + tstOff*numUnk],
																		bfRemapVec[gBas + basOff*numUnk],
																		elemMat(dim*iTst + tstOff, dim*iBas + basOff)));
					}
				}
			}
		}
	}
}

void LinearElasticProblem::setVector()
{
  pVVec = std::make_unique<EigenVector>(numFreeDOFs);
	pVVec->setZero();
	EigenVector& rVVec = *pVVec;
	for(auto cit = loadVec.begin(); cit != loadVec.end(); ++cit)
		rVVec(bfRemapVec[cit->first]) = cit->second;
}

std::pair<double, bool> LinearElasticProblem::computeCompliance()
{
	if(invalid)
	{
		std::cout << "Warning, FEM OFV not valid" << std::endl;
		return std::pair<double, bool>(1e6, false);
	}
	double sum = pForce->adjoint()*(*pVVec);
	return std::pair<double, bool>(sum, true);
}

double LinearElasticProblem::elementCompliance(const std::size_t kelem, const EigenDenseMat& elemMat, std::size_t numUnk) const
{
	if(dim == 2)
		return elementCompliance2D(kelem, elemMat, numUnk);
	return elementCompliance3D(kelem, elemMat, numUnk);
}

double LinearElasticProblem::elementCompliance2D(const std::size_t kelem, const EigenDenseMat& elemMat, std::size_t numUnk) const
{
	double res = 0.;
	const unsigned numElemNodes = probMesh->getNumElementNodes(kelem);
	for(unsigned iTst = 0; iTst < numElemNodes; iTst++)
	{
		std::size_t gTst = probMesh->getGlobalBF(kelem, iTst);
		for(unsigned iBas = 0; iBas < numElemNodes; iBas++)
		{
			std::size_t gBas = probMesh->getGlobalBF(kelem, iBas);
			for(unsigned tstdof = 0; tstdof < dim; ++tstdof)
			{
				std::size_t m = gTst + tstdof*numUnk;
				for(unsigned basdof = 0; basdof < dim; ++basdof)
				{
					std::size_t n = gBas + basdof*numUnk;
					if(!(fixedDOFs[m] || fixedDOFs[n]))
						res += elemMat(iTst + tstdof*numElemNodes, iBas + basdof*numElemNodes)*(*pVVec)(bfRemapVec[m])*(*pVVec)(bfRemapVec[n]);
				}
			}
		}
	}
	return res;
}

double LinearElasticProblem::elementCompliance3D(const std::size_t kelem, const EigenDenseMat& elemMat, std::size_t numUnk) const
{
	double res = 0.;
	const unsigned numElemNodes = probMesh->getNumElementNodes(kelem);
	for(unsigned iTst = 0; iTst < numElemNodes; iTst++)
	{
		std::size_t gTst = probMesh->getGlobalBF(kelem, iTst);
		for(unsigned iBas = 0; iBas < numElemNodes; iBas++)
		{
			std::size_t gBas = probMesh->getGlobalBF(kelem, iBas);
			for(unsigned tstOff = 0; tstOff < dim; ++tstOff)
			{
				std::size_t m = gTst + tstOff*numUnk;
				for(unsigned basOff = 0; basOff < dim; ++basOff)
				{
					std::size_t n = gBas + basOff*numUnk;
					if(!(fixedDOFs[m] || fixedDOFs[n]))
						res += elemMat(dim*iTst + tstOff, dim*iBas + basOff)*(*pVVec)(bfRemapVec[m])*(*pVVec)(bfRemapVec[n]);
				}
			}
		}
	}
	return res;
}

std::vector<double> LinearElasticProblem::gradCompliance(const Topologies::TOMesh &inMesh) const
{
    const std::size_t numNodes = probMesh->getNumUnknowns();
    const std::size_t nelems = probMesh->getNumElements();
    // Precompute uku values
    std::vector<double> ukuVec(nelems);
    for (std::size_t ielem = 0; ielem < nelems; ++ielem)
    {
        const Eigen::MatrixXd elemMat = probMesh->getElementMatrix(ielem);
        ukuVec[ielem] = -elementCompliance(ielem, elemMat, numNodes);
        ukuVec[ielem] /= inMesh.getOptVal(ielem);
    }
    return ukuVec;
}
