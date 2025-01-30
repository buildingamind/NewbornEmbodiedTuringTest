from nett import NETT
def main():
    NETT({
        'Brain': {
            'policy': 'CnnPolicy',
            'encoder': 'small',
            'algorithm': 'PPO'
        },
        'Body': {
        },
        'Environment': {
            'executable_path': 'path/to/executable.x86_64'
        },
        'Run': {
            'output_dir': 'path/to/output/dir/of/run/'
        }
    })

    NETT.run()

if __name__ == '__main__':  
    main()