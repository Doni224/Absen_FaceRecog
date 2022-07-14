-- phpMyAdmin SQL Dump
-- version 5.1.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jul 14, 2022 at 05:21 AM
-- Server version: 10.4.21-MariaDB
-- PHP Version: 8.0.10

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `bigprojeck`
--

-- --------------------------------------------------------

--
-- Table structure for table `jadwal`
--

CREATE TABLE `jadwal` (
  `pelajaran` varchar(50) NOT NULL,
  `dosen` varchar(50) NOT NULL,
  `waktu` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `jadwal`
--

INSERT INTO `jadwal` (`pelajaran`, `dosen`, `waktu`) VALUES
('Matematika', 'Hesti', '2021-12-06 05:25:25');

-- --------------------------------------------------------

--
-- Table structure for table `kehadiran`
--

CREATE TABLE `kehadiran` (
  `nama` varchar(60) NOT NULL,
  `waktu` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `kelasa`
--

CREATE TABLE `kelasa` (
  `nama` varchar(55) NOT NULL,
  `waktu` timestamp(6) NOT NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `kelasa`
--

INSERT INTO `kelasa` (`nama`, `waktu`) VALUES
('Abah', '2022-06-29 05:28:15.812422'),
('Bayas', '2022-06-29 05:31:04.614856');

-- --------------------------------------------------------

--
-- Table structure for table `kelasb`
--

CREATE TABLE `kelasb` (
  `nama` varchar(56) NOT NULL,
  `waktu` timestamp(6) NOT NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `kelasb`
--

INSERT INTO `kelasb` (`nama`, `waktu`) VALUES
('Bayas', '2022-06-29 05:36:15.245339');

-- --------------------------------------------------------

--
-- Table structure for table `kelasc`
--

CREATE TABLE `kelasc` (
  `nama` varchar(56) NOT NULL,
  `waktu` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `kelasc`
--

INSERT INTO `kelasc` (`nama`, `waktu`) VALUES
('Bayas', '2022-06-29 05:33:07');

-- --------------------------------------------------------

--
-- Table structure for table `kelasd`
--

CREATE TABLE `kelasd` (
  `nama` varchar(56) NOT NULL,
  `waktu` timestamp(6) NOT NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `kelasd`
--

INSERT INTO `kelasd` (`nama`, `waktu`) VALUES
('Bayas', '2022-06-30 14:05:30.180797');

-- --------------------------------------------------------

--
-- Table structure for table `kelase`
--

CREATE TABLE `kelase` (
  `nama` varchar(56) NOT NULL,
  `waktu` timestamp(6) NOT NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `user`
--

CREATE TABLE `user` (
  `nim` int(8) NOT NULL,
  `nama` varchar(50) NOT NULL,
  `email` varchar(50) NOT NULL,
  `password` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `user`
--

INSERT INTO `user` (`nim`, `nama`, `email`, `password`) VALUES
(12909929, 'adi', 'adii@gmail.com', ''),
(18090064, 'iran12', 'irpan1213213@gmail.com', '123'),
(18090099, 'irpan', 'ilham@gmail.com', '123456'),
(19090001, 'satu', 'arif123@gmail.com', '123456'),
(19090002, 'nopal', 'nopal@gmail.com', '123456'),
(19090003, 'agung', 'agung123@gmail.com', '123456'),
(19090010, 'arifullah', 'arif123@gmail.com', '123456'),
(19090064, 'ilham', 'ilham@gmail.com', '123456'),
(19090087, 'aaaa', 'ilham@gmail.com', '123456'),
(19090098, 'irvan ', 'irvan123@gmail.com', '123456'),
(19090099, 'Irvan Akbar Febriansyah', 'irvanfebriansyah21@gmail.com', '123456'),
(19090100, 'basudara', 'basudara123@gmail.com', '123'),
(19090140, 'elang', 'elang@gmail.com', '123456'),
(19092091, 'agung', 'adii@gmail.com', '123'),
(19099290, 'adi', 'adii@gmail.com', ''),
(190900246, 'Firman', 'firman15@gmail.com', '12345');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `kehadiran`
--
ALTER TABLE `kehadiran`
  ADD PRIMARY KEY (`nama`);

--
-- Indexes for table `user`
--
ALTER TABLE `user`
  ADD PRIMARY KEY (`nim`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
