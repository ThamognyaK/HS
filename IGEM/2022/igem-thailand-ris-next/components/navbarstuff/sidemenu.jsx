import { CloseIcon, HamburgerIcon } from '@chakra-ui/icons'
import { Box, Button, Slide, useDisclosure } from '@chakra-ui/react'
import ButtonU from './button'
import DropDowns from './dropdown'
import DropDownU from './dropdownside'
import Underline from './underline'
import ButtonS from './buttonsidebar'

const SideMenu = () => {
    const { isOpen, onToggle } = useDisclosure()
    return (
        <div className="grid grid-rows-1 grid-flow-col auto-cols-max justify-between text-xl">
            <ButtonU name="THAILAND_RIS" />
            <Button onClick={onToggle}>
                {isOpen ? <CloseIcon /> : <HamburgerIcon />}
            </Button>
            <Slide
                direction="left"
                in={isOpen}
                style={{
                    width: '300px',
                    zIndex: 0,
                    minHeight: '90vh',
                    padding: 0,
                    border: 0
                }}
            >
                <Box
                    p="0px"
                    borderRadius="0"
                    color="white"
                    mt="4"
                    bg="white"
                    textColor="black"
                    rounded="0"
                    shadow="md"
                    h="calc(96vh)"
                >
                    <div className="grid grid-cols-1">
                        <Box bg="#303f7b" p="4">
                            <div className="flex justify-between">
                                <img
                                    src="https://static.igem.wiki/teams/4314/wiki/igemlogo2.png"
                                    className="h-[80px] md:h-[100px]"
                                />
                                <Box px="3px"></Box>
                                <div className="self-center text-white text-[16px]">
                                    THAILAND_RIS
                                </div>
                            </div>
                        </Box>
                        <ButtonS name="HOME" />
                        <DropDownU
                            name="TEAM"
                            arr={['MEMBERS', 'ATTRIBUTIONS', 'COLLABORATIONS']}
                        />
                        <DropDownU
                            name="PROJECT"
                            arr={[
                                'DESCRIPTION',
                                'DESIGN',
                                'ENGINEERING',
                                'RESULTS',
                                'IMPLEMENTATION',
                                'CONTRIBUTION',
                                'NOTEBOOK',
                                'SAFETY'
                            ]}
                        />
                        <DropDownU
                            name="MODEL"
                            arr={['GENE SEQUENCE', 'CIRCULAR DNA']}
                        />
                        <DropDownU name="PARTS" arr={['PARTS COLLECTION']} />
                        <DropDownU
                            name="HUMAN PRACTICES"
                            arr={[
                                'INTEGRATED HUMAN PRACTICES',
                                'EDUCATIONAL OUTREACH',
                                'PATNERSHIP'
                            ]}
                        />
                        <ButtonS name="JUDGING FORM" />
                    </div>
                </Box>
            </Slide>
        </div>
    )
}

export default SideMenu
