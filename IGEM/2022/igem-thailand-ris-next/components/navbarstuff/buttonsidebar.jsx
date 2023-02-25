import { Button } from '@chakra-ui/react'
import Underline from './underline'

const ButtonS = props => {
    return (
        <>
            <div className="bg-white text-sm text-black">
                <Button
                    px={4}
                    py={2}
                    bg="white"
                    transition="all 0.2s"
                    border="0"
                    borderColor="white"
                    borderRadius="10"
                    borderWidth="0px"
                    boxShadow="none"
                    _hover={{ bg: 'gray.200' }}
                    _expanded={{ bg: 'gray.200' }}
                    _active={{ borderColor: 'gray.200' }}
                >
                    <div className="text-sm">
                        <Underline
                            hre={props.name.replace(/\s+/g, '-').toLowerCase()}
                            name={props.name}
                            color="black"
                        />
                    </div>
                </Button>
            </div>
        </>
    )
}

export default ButtonS
