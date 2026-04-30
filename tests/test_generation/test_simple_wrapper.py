from src.generation.simple_wrapper import simple_to_mps


def test_simple_generation_and_conversion():

    simple_root = 'models/sources/simple-methods'
    arguments = {"NBREGIONS": 4, "FROM": 0, "TO": 1, "METHOD": "standard_lp"}
    mps_output_dir = 'temp/mps'

    mps = simple_to_mps(simple_root=simple_root,
                        arguments=arguments,
                        mps_output_dir=mps_output_dir)

    print(mps)
