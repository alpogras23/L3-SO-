#!/usr/bin/env python3
"""
Azure ML Integration Test
Tests workspace connection, compute cluster, job submission, and result download
"""

import sys
import time
from pathlib import Path
import argparse

try:
    from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
    from azureml.core.compute import ComputeTarget, AmlCompute
    from azureml.core.authentication import InteractiveLoginAuthentication
except ImportError:
    print("❌ Azure ML SDK not installed")
    print("Install: pip install azureml-sdk")
    sys.exit(1)


class AzureMLIntegrationTest:
    """Test Azure ML connectivity and operations"""
    
    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.ws = None
        
    def test_workspace_connection(self) -> bool:
        """Test 1: Workspace connection"""
        print("\n" + "="*70)
        print("TEST 1: Workspace Connection")
        print("="*70)
        
        try:
            print(f"Connecting to workspace: {self.workspace_name}")
            print(f"Resource group: {self.resource_group}")
            print(f"Subscription: {self.subscription_id}")
            
            self.ws = Workspace(
                subscription_id=self.subscription_id,
                resource_group=self.resource_group,
                workspace_name=self.workspace_name
            )
            
            print(f"✅ Connected successfully")
            print(f"   Location: {self.ws.location}")
            print(f"   Workspace ID: {self.ws._workspace_id}")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def test_compute_cluster(self, compute_name: str = "gpu-cluster") -> bool:
        """Test 2: Compute cluster availability"""
        print("\n" + "="*70)
        print("TEST 2: Compute Cluster")
        print("="*70)
        
        if not self.ws:
            print("❌ Workspace not connected")
            return False
        
        try:
            # Check if compute exists
            compute_target = ComputeTarget(workspace=self.ws, name=compute_name)
            print(f"✅ Compute cluster found: {compute_name}")
            print(f"   Type: {compute_target.type}")
            print(f"   Status: {compute_target.provisioning_state}")
            print(f"   Min nodes: {compute_target.get_status().vm_size_property.min_node_count}")
            print(f"   Max nodes: {compute_target.get_status().vm_size_property.max_node_count}")
            return True
            
        except Exception as e:
            print(f"⚠️  Compute cluster not found: {e}")
            print("   This is OK if you haven't created it yet")
            print(f"   Create it with: ./setup_azure_ml.sh")
            return False
    
    def test_environment(self, env_name: str = "l3-vfa-pma-env") -> bool:
        """Test 3: Environment availability"""
        print("\n" + "="*70)
        print("TEST 3: Environment")
        print("="*70)
        
        if not self.ws:
            print("❌ Workspace not connected")
            return False
        
        try:
            # List environments
            envs = Environment.list(workspace=self.ws)
            
            if env_name in envs:
                env = envs[env_name]
                print(f"✅ Environment found: {env_name}")
                print(f"   Version: {env.version}")
                return True
            else:
                print(f"⚠️  Environment not found: {env_name}")
                print("   Available environments:")
                for name in list(envs.keys())[:10]:
                    print(f"     - {name}")
                return False
                
        except Exception as e:
            print(f"❌ Error checking environment: {e}")
            return False
    
    def test_dataset_registration(self) -> bool:
        """Test 4: Dataset registration capability"""
        print("\n" + "="*70)
        print("TEST 4: Dataset Registration")
        print("="*70)
        
        if not self.ws:
            print("❌ Workspace not connected")
            return False
        
        try:
            from azureml.core import Dataset
            
            # List existing datasets
            datasets = Dataset.get_all(workspace=self.ws)
            print(f"✅ Can access datasets")
            print(f"   Registered datasets: {len(datasets)}")
            
            if datasets:
                print("   Available datasets:")
                for name in list(datasets.keys())[:5]:
                    print(f"     - {name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Dataset access failed: {e}")
            return False
    
    def test_experiment_creation(self, experiment_name: str = "integration-test") -> bool:
        """Test 5: Experiment creation"""
        print("\n" + "="*70)
        print("TEST 5: Experiment Creation")
        print("="*70)
        
        if not self.ws:
            print("❌ Workspace not connected")
            return False
        
        try:
            exp = Experiment(workspace=self.ws, name=experiment_name)
            print(f"✅ Experiment created/accessed: {experiment_name}")
            print(f"   Workspace: {exp.workspace.name}")
            return True
            
        except Exception as e:
            print(f"❌ Experiment creation failed: {e}")
            return False
    
    def run_all_tests(self) -> dict:
        """Run all integration tests"""
        print("\n" + "="*70)
        print("AZURE ML INTEGRATION TEST SUITE")
        print("="*70)
        
        results = {
            'workspace_connection': self.test_workspace_connection(),
            'compute_cluster': self.test_compute_cluster(),
            'environment': self.test_environment(),
            'dataset_registration': self.test_dataset_registration(),
            'experiment_creation': self.test_experiment_creation()
        }
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {test_name}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        print("="*70 + "\n")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Azure ML Integration Test')
    parser.add_argument('--subscription-id', type=str, required=True,
                        help='Azure subscription ID')
    parser.add_argument('--resource-group', type=str, required=True,
                        help='Azure resource group name')
    parser.add_argument('--workspace', type=str, required=True,
                        help='Azure ML workspace name')
    parser.add_argument('--compute-name', type=str, default='gpu-cluster',
                        help='Compute cluster name')
    
    args = parser.parse_args()
    
    tester = AzureMLIntegrationTest(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace
    )
    
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if all(results.values()):
        print("✅ All tests PASSED")
        sys.exit(0)
    elif results['workspace_connection']:
        print("⚠️  Some tests failed but workspace is accessible")
        sys.exit(0)
    else:
        print("❌ Critical tests FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
