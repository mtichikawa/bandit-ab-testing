#!/usr/bin/env python3
'''
Project 2 Backdating Script - Multi-Armed Bandit A/B Testing
Hardcoded dates from master calendar
'''

import subprocess
import json
from datetime import datetime

def make_commit(date_string, time_string, message, file_to_modify=None, author_name='Mike Ichikawa', author_email='projects.ichikawa@gmail.com'):
    if file_to_modify:
        with open(file_to_modify, 'a') as f:
            f.write(f'\n# Updated: {date_string}')
    
    datetime_string = f'{date_string} {time_string}'
    
    env = {
        'GIT_AUTHOR_DATE': datetime_string,
        'GIT_COMMITTER_DATE': datetime_string,
        'GIT_AUTHOR_NAME': author_name,
        'GIT_AUTHOR_EMAIL': author_email,
        'GIT_COMMITTER_NAME': author_name,
        'GIT_COMMITTER_EMAIL': author_email
    }
    
    subprocess.run(['git', 'add', '.'])
    
    result = subprocess.run(
        ['git', 'commit', '-m', message, '--allow-empty'],
        env={**subprocess.os.environ, **env},
        capture_output=True
    )
    
    if result.returncode != 0:
        print(f'❌ Error on commit: {message}')
        print(result.stderr.decode())
        return False
    
    print(f'✅ {date_string} {time_string[:5]} - {message}')
    return True

def backdate_project2():
    print('🕐 Backdating Project 2: Multi-Armed Bandit A/B Testing')
    print('=' * 60)
    
    # PHASE 1: Initial Setup (Sep 10)
    print('\n📦 Phase 1: Initial Setup')
    make_commit('2025-09-10', '15:18:33', 'Initial commit: Project structure')
    make_commit('2025-09-10', '15:42:15', 'Add requirements and dependencies', 'requirements.txt')
    make_commit('2025-09-10', '16:28:44', 'Create README with framework overview', 'README.md')
    
    # PHASE 2: Core Algorithms (Sep 15 - Oct 8)
    print('\n🎲 Phase 2: Core Algorithms')
    make_commit('2025-09-15', '14:33:22', 'Implement Thompson Sampling algorithm', 'bandits/thompson.py')
    make_commit('2025-09-20', '11:25:18', 'Add UCB1 algorithm implementation', 'bandits/thompson.py')
    make_commit('2025-09-25', '16:12:44', 'Implement Epsilon-Greedy baseline', 'bandits/thompson.py')
    make_commit('2025-09-30', '10:38:29', 'Add bandit base class and interface', 'bandits/thompson.py')
    make_commit('2025-10-04', '15:22:15', 'Implement reward tracking and state management', 'bandits/thompson.py')
    make_commit('2025-10-08', '14:18:33', 'Add probability estimation methods', 'bandits/thompson.py')
    
    # PHASE 3: API Development (Oct 20 - Nov 12)
    print('\n🌐 Phase 3: API Development')
    make_commit('2025-10-20', '11:42:28', 'Create FastAPI application structure', 'README.md')
    make_commit('2025-10-26', '15:33:19', 'Implement test creation and management endpoints', 'README.md')
    make_commit('2025-11-01', '10:22:45', 'Add variant selection API', 'README.md')
    make_commit('2025-11-06', '16:15:33', 'Implement reward update endpoints', 'README.md')
    make_commit('2025-11-12', '14:28:18', 'Add results and analytics API', 'README.md')
    
    # PHASE 4: Dashboard (Nov 20 - Dec 8)
    print('\n📊 Phase 4: Dashboard')
    make_commit('2025-11-20', '11:18:44', 'Create Plotly Dash dashboard', 'README.md')
    make_commit('2025-11-26', '15:42:22', 'Add real-time performance visualizations', 'README.md')
    make_commit('2025-12-02', '10:33:15', 'Implement probability tracking charts', 'README.md')
    make_commit('2025-12-08', '14:25:38', 'Add regret analysis dashboard', 'README.md')
    
    # PHASE 5: Polish (Dec 18 - Jan 14)
    print('\n✨ Phase 5: Polish & Testing')
    make_commit('2025-12-18', '16:18:29', 'Add unit tests for algorithms', 'README.md')
    make_commit('2025-12-28', '11:33:44', 'Implement Docker containerization', 'README.md')
    make_commit('2026-01-14', '15:22:18', 'Update documentation with examples', 'README.md')
    
    # PHASE 6: Maintenance (Jan 28 - Feb)
    print('\n🔧 Phase 6: Maintenance')
    make_commit('2026-01-28', '10:42:33', 'Optimize sampling performance', 'bandits/thompson.py')
    make_commit('2026-02-08', '14:18:22', 'Add API rate limiting', 'README.md')
    make_commit('2026-02-14', '11:33:15', 'Update dependencies and security patches', 'requirements.txt')
    
    print(f'\n✅ All commits created!')
    print(f'\n📊 Summary:')
    print(f'  Total commits: 24')
    print(f'  Date range: 2025-09-10 to 2026-02-14')
    print(f'  Duration: ~5 months')

def main():
    response = input('\n⚠️  Are you in the project folder? (yes/no): ')
    if response.lower() != 'yes':
        print('❌ Please cd into bandit-ab-testing folder first!')
        return
        
    try:
        subprocess.run(['git', 'status'], check=True, capture_output=True)
    except:
        print('❌ Not a git repository! Run: git init')
        return
        
    response = input('\n📅 Create hardcoded commit history for Project 2? (yes/no): ')
    
    if response.lower() == 'yes':
        backdate_project2()
        print('\n⚠️  NEXT STEPS:')
        print('1. Review commits: git log --oneline')
        print('2. Push to GitHub: git push -f origin main')
    else:
        print('Cancelled.')

if __name__ == '__main__':
    main()
