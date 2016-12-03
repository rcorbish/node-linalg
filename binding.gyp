{
  "targets": [
{
      "target_name": "lalg",
      "sources": [ "src/Array.cpp" ], 
      "defines" : [
	"EIGEN_MPL2_ONLY"
	],
      "libraries": [
            "-lopenblas", "-lpthread", "-lgfortran", "-llapacke"
        ],
      'include_dirs': [ 
		'eigen' , 
		'CppNumericalSolvers/include' 
      ],
      "cflags": ["-Wall", "-std=c++11", "-Wsign-compare" ],
      'xcode_settings': {
        'OTHER_CFLAGS': [
          '-std=c++11'
        ],
      },
      "conditions": [
        [ 'OS=="mac"', {
            "xcode_settings": {
                'OTHER_CPLUSPLUSFLAGS' : ['-std=c++11','-stdlib=libc++'],
                'OTHER_LDFLAGS': ['-stdlib=libc++'],
                'MACOSX_DEPLOYMENT_TARGET': '10.7' }
            }
        ]
      ]
}
  ]
}

